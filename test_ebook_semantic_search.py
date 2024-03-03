import unittest
import ebook_semantic_search as ess
from os.path import exists


class TestSemanticSearch(unittest.TestCase):

    def setUp(self):
        # Set up any necessary variables or configurations for testing
        # noinspection SpellCheckingInspection
        self._json_path = \
            r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).json'

        # Delete the existing json file if it exists
        if exists(self._json_path):
            import os
            os.remove(self._json_path)

        # Check if the JSON file exists, if not, load EPUB and save embeddings in JSON
        if not exists(self._json_path):
            # Generate EPUB file path by replacing .json with .epub
            epub_path = self._json_path.replace('.json', '.epub')
            search_instance = ess.SemanticSearch()
            search_instance.load_file(epub_path)

    def test_query(self):
        # Test loading an EPUB file
        search_instance = ess.SemanticSearch()
        search_instance.load_file(self._json_path)

        # Define your query and expected output
        query = 'Why do we need to corroborate theories at all?'
        expected_results = [501, 441, 462, 465, 122]
        expected_results_msgs = '''
            Chapter: "10 CORROBORATION, OR HOW A THEORY STANDS UP TO TESTS", Passage number: 60, Score: 0.69
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
            etc.’ But the question is not one of expediency—it is one of scientific curiosity. Some scientists want
            to explain the world: their aim is to find satisfactory explanatory theories—well testable, i.e. simple
            theories—and to test them. (See also appendix *x and section *15 of my Postscript.)"
            '''
        # Call the search method and get the actual results
        actual_results_msgs, actual_results = search_instance.search(query, top_results=5)
        stripped_result_actual = (actual_results_msgs[0].replace(" ", "").replace("\t", "")
                                  .replace("\n", ""))
        stripped_expected = (expected_results_msgs.replace(" ", "").replace("\t", "")
                             .replace("\n", ""))
        self.assertEqual(expected_results, actual_results)
        self.assertEqual(stripped_expected, stripped_result_actual)

    def tearDown(self):
        # Clean up any resources or configurations after testing
        pass
