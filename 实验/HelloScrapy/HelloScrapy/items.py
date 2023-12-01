# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MovieItem(scrapy.Item):
    title = scrapy.Field()
    rating_num = scrapy.Field()
    inq = scrapy.Field()
    comment = scrapy.Field()


