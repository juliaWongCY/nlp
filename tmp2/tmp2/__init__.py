import time
from pyramid.config import Configurator
from pyramid.static import QueryStringConstantCacheBuster


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings)
    config.include('pyramid_jinja2')
    config.include('pyramid_beaker')
    config.include('pyramid_excel')
    config.add_static_view('static', 'static')
    config.add_cache_buster(
        'static',
        QueryStringConstantCacheBuster(str(int(time.time()))))
    config.add_route('home', '/')
    config.add_route('score', '/score')
    config.add_route('result', '/score/result')
    config.add_route('option', '/score/option')
    config.add_route('statistics', '/statistics')
    config.add_route('re_statistics', '/statistics/refresh')
    config.add_route('analysis', '/analysis')
    config.scan()
    return config.make_wsgi_app()
