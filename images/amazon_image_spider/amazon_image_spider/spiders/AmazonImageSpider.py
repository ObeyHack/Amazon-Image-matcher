import scrapy

class AmazonImageSpider(scrapy.Spider):
    name = 'amazon_image_spider'
    with open('urls.txt') as f:
        start_urls = [url.strip() for url in f.readlines()]

    def parse(self, response):
        yield {
            'image_urls': response.url
        }

