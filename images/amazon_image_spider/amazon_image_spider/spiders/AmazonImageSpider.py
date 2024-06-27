import scrapy
import sys
sys.path.append('..')
from items import AmazonImageSpiderItem
from scrapy import cmdline


class AmazonImageSpider(scrapy.Spider):
    name = 'amazon_image_spider'
    with open('../../urls.txt') as f:
        start_urls = [url.strip() for url in f.readlines()]

    def parse(self, response):
        item = AmazonImageSpiderItem()
        item['image_urls'] = response.url
        yield item


if __name__ == '__main__':
    cmdline.execute("scrapy runspider AmazonImageSpider.py".split())