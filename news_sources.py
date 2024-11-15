# Dictionary of news sources with their URLs and CSS selectors for article extraction
NEWS_SOURCES = {
    'Yahoo Finance': {
        'url': 'https://finance.yahoo.com/quote/{}/news',
        'selectors': [
            'h3.Mb\\(5px\\) a',  # Headlines in main news section
            'a[data-test="quoteNewsLink"]'  # News links
        ]
    },
    'Reuters': {
        'url': 'https://www.reuters.com/search/news?blob={}',
        'selectors': [
            'h3.search-result-title a',
            'div.story-content a'
        ]
    },
    'MarketWatch': {
        'url': 'https://www.marketwatch.com/search?q={}&m=Keyword',
        'selectors': [
            'div.article__content a',
            'h3.article__headline a'
        ]
    },
    'Financial Times': {
        'url': 'https://www.ft.com/search?q={}',
        'selectors': [
            'div.o-teaser__heading a',
            'h3.o-teaser__heading a'
        ]
    }
}
