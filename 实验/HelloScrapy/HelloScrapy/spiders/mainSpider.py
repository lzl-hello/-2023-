import scrapy
from scrapy import Selector, Request

from HelloScrapy.items import MovieItem


class MainspiderSpider(scrapy.Spider):
    name = "mainSpider"
    allowed_domains = ["movie.douban.com"]
    start_urls = ["https://movie.douban.com/top250"]

    def parse(self, response, **kwargs):
        sel = Selector(response)
        list_items = sel.css('#content > div > div.article > ol > li')
        for list_item in list_items[:10]:
            detail_url = list_item.css('div.info > div.hd > a::attr(href)').extract_first()
            # print(detail_info)
            movie_item = MovieItem()
            movie_item['title'] = list_item.css('span.title::text').extract_first()
            movie_item['rating_num'] = list_item.css('span.rating_num::text').extract_first()
            movie_item['inq'] = list_item.css('span.inq::text').extract_first()
            yield Request(url=detail_url, callback=self.parse_detail, cb_kwargs={'item':movie_item})

    def parse_detail(self, response, **kwargs):
        movie_item = kwargs['item']
        sel = Selector(response)
        # comment_items = sel.css('#hot-comments > div')
        # for comment_item in comment_items[:5]:
        #     # movie_item['comment'] += comment_item.css('span.short::text').extract_first()
        #     comment = comment_item.css('span.short::text').extract_first()
        #     print(comment)
        movie_item['comment'] = sel.css('span.short::text').extract()

        yield movie_item
