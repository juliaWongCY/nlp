import unittest

from pyramid import testing


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.config = testing.setUp()

    def tearDown(self):
        testing.tearDown()

    def test_my_view(self):
        from .views import my_view
        request = testing.DummyRequest()
        info = my_view(request)
        self.assertEqual(info['project'], 'NPS analyser')

    # def test_score_view(self):
    #     from .views import score_view
    #     request = testing.DummyRequest()
    #     response = score_view(request)
    #     expected_keys = ['score', 'class']
    #
    #     self.assertCountEqual(list(response.keys()), expected_keys)


class FunctionalTests(unittest.TestCase):
    def setUp(self):
        from tmp2 import main
        app = main({})
        from webtest import TestApp
        self.testapp = TestApp(app)

    # def test_root(self):
    #     res = self.testapp.get('/', status=200)
    #     self.assertTrue(b'Project Group 6' in res.body)
