from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect
from .models import Article
from .search_engine import SearchEngine
from .forms import SearchForm


search_engine = SearchEngine(resources_path='./search_engine/resources_s/')


def index(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('search/str=' + str(form.data['search_string']) + '/')
    else:
        form = SearchForm()

    return render(request, 'search_engine/index.html', {'form': form})


def search(request, search_str):
    res_list = search_engine.search([search_str])
    if len(res_list) > 10:
        res_list = res_list[:10]
    found_articles_list = []
    for result in res_list:
        found_articles_list.append(Article.objects.get(title__exact=result))
    context = {'found_articles_list': found_articles_list}
    return render(request, 'search_engine/search.html', context)


def detail(request, article_id):
    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'search_engine/detail.html', {'article': article})
