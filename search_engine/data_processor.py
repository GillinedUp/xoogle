from search_engine.models import Article
from search_engine.search_engine import load_list


def fill_data(res_path, art_path):
    file_list = load_list(res_path, 'file_list')
    doc_list = []
    for file in file_list:
        with open(art_path + file, 'r') as myfile:
            doc_list.append(myfile.read())
    for i in range(0, len(file_list)):
        article = Article.objects.create_article(file_list[i], doc_list[i])
        article.save()
