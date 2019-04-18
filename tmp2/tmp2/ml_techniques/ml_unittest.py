import unittest

from pyramid import testing


class MLUnitTest(unittest.TestCase):
    def setUp(self):
        self.config = testing.setUp()

    def tearDown(self):
        testing.tearDown()

    def test_ml_name(self):
        from .ml_technique import MLTechnique
        from sklearn.tree import tree
        ml_test = MLTechnique(name='test technique', data_file='dummy.xlsx', model=tree.DecisionTreeClassifier(),
                              features=['neu', 'neg', 'pos', 'compound'])
        self.assertEqual(ml_test.get_name(), 'test technique')
