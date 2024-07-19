import glob
import random
import time
import zipfile
from urllib.request import urlopen, build_opener
import pandas as pd
import requests
from bs4 import BeautifulSoup
import torch
from faker import Faker
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def text_embedding(text):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Get the BERT model embeddings
    with torch.no_grad():
        model_output = model(input_ids)

    # Extract the embeddings from the model output
    embeddings = model_output.last_hidden_state

    # Calculate the average of the embeddings for each token
    avg_embeddings = torch.mean(embeddings, dim=1)

    return avg_embeddings


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

    # opener = build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # time.sleep(0.5 * random.random())
    #
    # response = opener.open(url)
    # html = response.read()
    #
    # # Parse the HTML content using BeautifulSoup
    # soup = BeautifulSoup(html, 'html.parser')
    soup = get_soup_retry(url)

    # Find the element with id="productDescription"
    description_div = soup.find(id='productDescription')

    if description_div:
        # Extract the text within the <p><span> tags
        description_text = description_div.find('p').find('span').get_text()
    else:
        print("Description not found")
        description_text = ""

    # Find the title of the product
    print(url)
    title = soup.select_one('#productTitle').text

    return title+description_text


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
