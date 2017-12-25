from django.db import models


class ArticleManager(models.Manager):
    def create_article(self, title, text):
        article = self.create(title=title, text=text)
        # do something with the book
        return article


class Article(models.Model):
    title = models.CharField(max_length=100)
    text = models.TextField()
    objects = ArticleManager()

    def __str__(self):
        return self.title
