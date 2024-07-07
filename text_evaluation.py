from urllib.request import urlopen
from bs4 import BeautifulSoup

import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

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


def description_scraper(url):
    response = urlopen(url)
    html = response.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the element with id="productDescription"
    description_div = soup.find(id='productDescription')

    if description_div:
        # Extract the text within the <p><span> tags
        description_text = description_div.find('p').find('span').get_text()
        print(description_text)
    else:
        print("Description not found")
    return description_text


def similarity_score(url1, url2):
    text1 = text_embedding(description_scraper(url1))
    text2 = text_embedding(description_scraper(url2))
    similarity = cosine_similarity(text1, text2)
    return similarity


if __name__ == '__main__':
    url1 = 'https://www.amazon.com/Under-Armour-Charged-Assert-X-Wide/dp/B08CFT75X3/ref=sr_1_2?_encoding=UTF8&content-id=amzn1.sym.56e14e61-447a-443b-9528-4b285fddeeac&crid=1QEZIUFPCL3YZ&dib=eyJ2IjoiMSJ9.ft2_UOW6_812lc9l1-QSVp262n9lnrp9JkYxbzch50YDBc3lzBNyzMAiBk-I0IdyUcrfaGVjLJRshNC2heUyGwkRM8s0DoTb4M6iESi81wnkVgmzqAjgcRlkbEfcDI24cTaNoVMc3Mdool0oekYx_66W7cs9xa5ygzH_QQjvrB0aNX-Mz-IKmLBuA6CGzSxzDgw_WbXkr6Xhdj7AwUuSIj9YhQVnyp4PvUZ3YtcB7qdUQcQHrIv325on_XbSy7GY5SU2aZGHOTLcpAiBLoyJGZCQLeNUz3abwIVYKtMoNGI.ThotIlFS47Lro8cttfDqEFWQr5sueLmTYX1UdYgp-yg&dib_tag=se&keywords=Shoes&pd_rd_r=20072d94-7d9c-4817-9c8a-c541f1ee3e84&pd_rd_w=iiWNK&pd_rd_wg=UgsLQ&pf_rd_p=56e14e61-447a-443b-9528-4b285fddeeac&pf_rd_r=C7WVXH61VXFF81AV2G2Y&qid=1720343515&refinements=p_36%3A-5000&rnid=2661611011&sprefix=shoes%2Caps%2C145&sr=8-2&th=1'
    url2='https://www.amazon.com/dp/B0BZXMXTT2/ref=sspa_dk_detail_1?psc=1&pf_rd_p=386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_r=FND5XJ34Y17881CR2G9D&pd_rd_wg=Yvcge&pd_rd_w=cexMT&content-id=amzn1.sym.386c274b-4bfe-4421-9052-a1a56db557ab&pd_rd_r=c790c2c3-81de-4a2c-b1a0-9b79447aab15&s=shoes&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM'
    print(similarity_score(url1, url2))
