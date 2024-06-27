
from scrapy import cmdline


if __name__ == '__main__':
    cmdline.execute("scrapy runspider amazon_image_spider/spiders/AmazonImageSpider.py".split())