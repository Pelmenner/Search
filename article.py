class Article:
    def __init__(self, title: str, author: str, text: str, popularity: int, rating: int, article_id: int):
        self.title = title
        self.author = author
        self.text = text
        self.popularity = popularity
        self.rating = rating
        self.id = article_id

    @property
    def url(self):
        return f'https://habr.com/ru/post/{self.id}/'

    def format(self, query):
        return [self.title, self.text[:50] + ' ...']