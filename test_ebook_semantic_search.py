import unittest
import ebook_semantic_search as ess
from os.path import exists


class TestEpubSemanticSearch(unittest.TestCase):
    def setUp(self):
        self._instance = ess.SemanticSearch()
        # Set up any necessary variables or configurations for testing
        # noinspection SpellCheckingInspection
        self._book_path = \
            r'D:\Documents\Books\Karl Popper - The Logic of Scientific Discovery-Routledge (2002)(epub).epub'
        self._json_path = self._instance.switch_ext(self._book_path, '.json')

        # Delete the existing json file if it exists
        if exists(self._json_path):
            import os
            os.remove(self._json_path)
        self._instance.load_file(self._book_path)

    def test_epub(self):
        # Define your query and expected output
        query = 'Why do we need to corroborate theories at all?'
        expected_results = [643, 593, 568, 597, 622]
        expected_results_msgs = '''
            Chapter: "10 CORROBORATION, OR HOW A THEORY STANDS UP TO TESTS", Passage number: 75, Score: 0.69
            "*6 See my Postscript, chapter *ii. In my theory of corroboration—in direct opposition to Keynes’s, 
            Jeffreys’s, and Carnap’s theories of probability—corroboration does not decrease with testability, but 
            tends to increase with it.
            *7 This may also be expressed by the unacceptable rule: ‘Always choose the hypothesis which is most ad hoc!’
            2 Keynes, op. cit., p. 305.
            *8 Carnap, in his Logical Foundations of Probability, 1950, believes in the practical value of predictions; 
            nevertheless, he draws part of the conclusion here mentioned—that we might be content with our basic 
            statements. For he says that theories (he speaks of ‘laws’) are ‘not indispensable’ for science—not even 
            for making predictions: we can manage throughout with singular statements. ‘Nevertheless’, he writes 
            (p. 575) ‘it is expedient, of course, to state universal laws in books on physics, biology, psychology, 
            etc.’ But the question is not one of expediency—it is one of scientific curiosity. Some scientists want to 
            explain the world: their aim is to find satisfactory explanatory theories—well testable, i.e. simple 
            theories—and to test them. (See also appendix *x and section *15 of my Postscript.)"
            '''
        # Call the search method and get the actual results
        actual_results_msgs, actual_results = self._instance.search(query, top_results=5)
        stripped_result_actual = (actual_results_msgs[0].replace(" ", "").replace("\t", "")
                                  .replace("\n", ""))
        stripped_expected = (expected_results_msgs.replace(" ", "").replace("\t", "")
                             .replace("\n", ""))
        self.assertEqual(expected_results, actual_results)
        self.assertEqual(stripped_expected, stripped_result_actual)

    def tearDown(self):
        # Delete the existing json file if it exists
        if exists(self._json_path):
            import os
            os.remove(self._json_path)


class TestPdfSemanticSearch(unittest.TestCase):
    def setUp(self):
        self._instance = ess.SemanticSearch()
        # Set up any necessary variables or configurations for testing
        # noinspection SpellCheckingInspection
        self._book_path = \
            r'D:\Documents\Books\Karl Popper - The Logic of Scientific Discovery-Routledge (2002)(pdf).pdf'
        self._json_path = self._instance.switch_ext(self._book_path, '.json')

        # Delete the existing json file if it exists
        if exists(self._json_path):
            import os
            os.remove(self._json_path)
        self._instance.load_file(self._book_path)
        super().setUp()

    def test_pdf(self):
        # Define your query and expected output
        query = 'Why do we need to corroborate theories at all?'
        expected_results = [713, 734, 720, 105, 743]
        expected_results_msgs = '''
            Page number: 287, Passage number: 713, Score: 0.69
            "We say that a theory is ‘corroborated’ so long as it stands up to these tests. The appraisal which asserts 
            corroboration (the corroborativeappraisal) establishes certain fundamental relations, viz. compatibility 
            and incompatibility. We regard incompatibility as falsi fication of the theory. But compatibility alone 
            must not make us attribute to the theory a positive degree of corroboration: the mere fact that a theory 
            hasnot yet been falsi fied can obviously not be regarded as su fficient. For nothing is easier than to 
            construct any number of theoretical systemssome structural components of a theory of experience 264
            which are compatible with any given system of accepted basic statements. (This remark applies also to all 
            ‘metaphysical’ systems.)"
            '''
        # Call the search method and get the actual results
        actual_results_msgs, actual_results = self._instance.search(query, top_results=5)
        stripped_result_actual = (actual_results_msgs[0].replace(" ", "").replace("\t", "")
                                  .replace("\n", ""))
        stripped_expected = (expected_results_msgs.replace(" ", "").replace("\t", "")
                             .replace("\n", ""))
        self.assertEqual(expected_results, actual_results)
        self.assertEqual(stripped_expected, stripped_result_actual)

    def tearDown(self):
        # Delete the existing json file if it exists
        if exists(self._json_path):
            import os
            os.remove(self._json_path)
        super().tearDown()
