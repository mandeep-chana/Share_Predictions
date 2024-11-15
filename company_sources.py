def get_clean_ticker(ticker):
    """Remove exchange suffixes and clean ticker symbol for web searches"""
    # Remove common exchange suffixes
    suffixes = ['.L', '.TO', '.SA', '.F', '.PA', '.AS', '.BR', '.LS', '.MC', '.MI',
                '.VI', '.ST', '.OL', '.CO', '.HE', '.IS', '.WA', '.AT', '.T', '.HK',
                '.SS', '.SZ', '.KS', '.TW', '.BO', '.NS', '.AX', '.NZ', '.SI', '.JO', '.TA']

    clean_ticker = ticker
    for suffix in suffixes:
        if clean_ticker.endswith(suffix):
            clean_ticker = clean_ticker[:-len(suffix)]
            break

    return clean_ticker


# Dictionary of company information sources with their URLs and CSS selectors
COMPANY_SOURCES = {
    'Yahoo Finance': {
        'url': 'https://finance.yahoo.com/quote/{}',
        'selectors': {
            'description': 'section[data-test="qsp-profile"] p',
            'sector': 'span[data-test="qsp-profile-sector"]',
            'industry': 'span[data-test="qsp-profile-industry"]',
            'employees': 'span[data-test="qsp-profile-employees"]'
        }
    },
    'MarketWatch': {
        'url': 'https://www.marketwatch.com/investing/stock/{}',
        'selectors': {
            'description': 'div.description__text',
            'sector': 'div.company__metadata span:nth-child(1)',
            'industry': 'div.company__metadata span:nth-child(2)'
        }
    },
    'Reuters': {
        'url': 'https://www.reuters.com/companies/{}',
        'selectors': {
            'description': 'div.ProfileSummary-description',
            'sector': 'div.ProfileSummary-data-item:nth-child(1) span',
            'industry': 'div.ProfileSummary-data-item:nth-child(2) span'
        }
    }
}
