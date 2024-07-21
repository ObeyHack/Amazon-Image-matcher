import zipfile
import pandas as pd
import requests
from bs4 import BeautifulSoup
import torch
from faker import Faker
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')


tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16)


def tokenize(text):
    stopword = set(stopwords.words('english'))
    symbols = ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%',
               '^', '&', '*', '-', '+', '=', '_', '~', '`', '\'', '\"', '“', '”', '’', '‘', '\n', '\t', '–', '—', '•',]
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stopword]
    text = [word for word in text if word not in symbols]
    text = ' '.join(text)
    return text


def text_embedding(text):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Encode the input text
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def jaccard_similarity(doc1, doc2):
    # Tokenize the documents
    words_doc1 = set(doc1.split())
    words_doc2 = set(doc2.split())

    # Calculate intersection and union
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity
    similarity = len(intersection) / len(union)
    return similarity


def get_soup_retry(url, verbose=False):
    fake = Faker()
    uag_random = fake.user_agent()

    header = {
        'User-Agent': uag_random,
        'Accept-Language': 'en-US,en;q=0.9'
    }
    isCaptcha = True
    while isCaptcha:
        page = requests.get(url, headers=header)
        assert page.status_code == 200
        soup = BeautifulSoup(page.content, 'lxml')
        if 'captcha' in str(soup):
            uag_random = fake.user_agent()
            if verbose:
                print(f'\rBot has been detected... retrying ... use new identity: {uag_random} ', end='', flush=True)
            continue
        else:
            if verbose:
                print('Bot bypassed')
            return soup


def description_scraper(url):
    soup = get_soup_retry(url)

    # Find the element with id="productDescription"
    description_div = soup.find(id='productDescription')
    description_div2 = soup.find(id='productDescription_fullView')

    description_text1 = description_div.text if description_div else ""
    description_text2 = description_div2.text if description_div2 else ""
    description_text = description_text1 + description_text2

    title = soup.find(id='title').text
    return tokenize(title+description_text)


def similarity_score(url1, url2):
    text1 = text_embedding(description_scraper(url1))
    text2 = text_embedding(description_scraper(url2))
    similarity = cosine_similarity(text1, text2)
    return similarity


def similar_links(text, urls):
    similarities = []
    text_emb = text_embedding(text)
    for link in tqdm(urls):
        description = description_scraper(link)
        similarity = cosine_similarity(text_emb, text_embedding(description))
        similarities.append((link, similarity))
    return similarities


def get_urls():
    with zipfile.ZipFile("Datasets/archive.zip", 'r') as zip_ref:
        zip_ref.extractall("Datasets")

    df = pd.read_csv("Datasets/Amazon-Products.csv")
    urls = df['link']
    return urls


if __name__ == '__main__':
    # urls = get_urls()
    # text = "A stylish and comfortable pair of shoes for everyday wear"
    # similarities = similar_links(text, urls)

    url1 = 'https://www.amazon.com/Under-Armour-Charged-Assert-X-Wide/dp/B08CFT75X3/ref=sr_1_2?_encoding=UTF8&content-id=amzn1.sym.56e14e61-447a-443b-9528-4b285fddeeac&crid=1QEZIUFPCL3YZ&dib=eyJ2IjoiMSJ9.ft2_UOW6_812lc9l1-QSVp262n9lnrp9JkYxbzch50YDBc3lzBNyzMAiBk-I0IdyUcrfaGVjLJRshNC2heUyGwkRM8s0DoTb4M6iESi81wnkVgmzqAjgcRlkbEfcDI24cTaNoVMc3Mdool0oekYx_66W7cs9xa5ygzH_QQjvrB0aNX-Mz-IKmLBuA6CGzSxzDgw_WbXkr6Xhdj7AwUuSIj9YhQVnyp4PvUZ3YtcB7qdUQcQHrIv325on_XbSy7GY5SU2aZGHOTLcpAiBLoyJGZCQLeNUz3abwIVYKtMoNGI.ThotIlFS47Lro8cttfDqEFWQr5sueLmTYX1UdYgp-yg&dib_tag=se&keywords=Shoes&pd_rd_r=20072d94-7d9c-4817-9c8a-c541f1ee3e84&pd_rd_w=iiWNK&pd_rd_wg=UgsLQ&pf_rd_p=56e14e61-447a-443b-9528-4b285fddeeac&pf_rd_r=C7WVXH61VXFF81AV2G2Y&qid=1720343515&refinements=p_36%3A-5000&rnid=2661611011&sprefix=shoes%2Caps%2C145&sr=8-2&th=1'
    url2 = 'https://www.amazon.com/dp/B0BZXMXTT2/ref=sspa_dk_detail_1?psc=1&pf_rd_p=386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_r=FND5XJ34Y17881CR2G9D&pd_rd_wg=Yvcge&pd_rd_w=cexMT&content-id=amzn1.sym.386c274b-4bfe-4421-9052-a1a56db557ab&pd_rd_r=c790c2c3-81de-4a2c-b1a0-9b79447aab15&s=shoes&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM'
    print(similarity_score(url1, url2))
    # print(similar_links(text, urls))
    url3 = 'https://www.amazon.com/LEGO-Disney-Stitch-Building-Buildable/dp/B0CGY26D8G/ref=sr_1_2?crid=23QXV07HHVE7M&dib=eyJ2IjoiMSJ9.L1iHMZSfL_eRoYJJ69o-g2IWQlmfgJkyM2LBjhLKlvsmkzIA9Zh2e4QSKHALLuqwy1d2M_ESlzhsDcpjIh7pq_CZrm5-Zb2agU1r-sZNGxEioi8YWdvV2hBLeNCAjXJ2y91k2g08MsLNkkRiJoKTQkElGXyay7_2d-qJFGOyIz2l5lJ_QgkjW-B_i0HbcYyeOjhVguf03Rgkps7ORX4S_CXTnHCTCJHwEp__yG9gxVoNmi5M7F0I6WmVvgcswDWOD5VcZOwIuM6bgp2Wo9QO9rABEvAfiqxWOgJL7hJpTkk.7Z34_veo1afRteTuAz4oI6qG5RDHZKAyH0EXoDFmrHU&dib_tag=se&keywords=lego&qid=1721051615&sprefix=lego%2B%2Caps%2C222&sr=8-2&th=1'
    # print(similarity_score(url1, url2))
    print(similarity_score(url1, url3))
