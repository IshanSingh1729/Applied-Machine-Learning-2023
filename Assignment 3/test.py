import unittest
from score import score
from sklearn.neural_network import MLPClassifier
import joblib



class TestScore(unittest.TestCase):
    
    def setUp(self):
        # Load test data
        self.test_spam = "You have won $100,000 in this month's lottery. Reply with your credit card information to proceed."
        self.test_non_spam = "Hi, how are you doing today?"
        self.test_text = "This is a positive text"
        self.threshold = 0.5
        self.model = joblib.load('model/multilayer-perceptron-model/model.pkl')
        
    def test_smoke(self):
        # Test if the function produces some output without crashing
        prediction, propensity = score(self.test_text, self.model, self.threshold)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)
    
    def test_format(self):
        # Test if the input/output formats/types are as expected
        prediction, propensity = score(self.test_text, self.model, self.threshold)
        self.assertIsInstance(prediction, int)
        self.assertIsInstance(propensity, float)
    
    def test_prediction_value(self):
        # Test if the prediction value is 0 or 1
        prediction, propensity = score(self.test_text, self.model, self.threshold)
        self.assertIn(prediction, [0, 1])
    
    def test_propensity_score(self):
        # Test if the propensity score is between 0 and 1
        prediction, propensity = score(self.test_text, self.model, self.threshold)
        self.assertGreaterEqual(propensity, 0)
        self.assertLessEqual(propensity, 1)
    
    def test_threshold_zero(self):
        # Test if the prediction is always 1 when the threshold is 0
        prediction, propensity = score(self.test_text, self.model, 0)
        self.assertEqual(prediction, 1)
        
    def test_threshold_one(self):
        # Test if the prediction is always 0 when the threshold is 1
        prediction, propensity = score(self.test_text, self.model, 1)
        self.assertEqual(prediction, 0)
    
    def test_obvious_spam(self):
        # Test if the prediction is 1 on an obvious spam input text
        prediction, propensity = score(self.test_spam, self.model, self.threshold)
        self.assertEqual(prediction, 1)
        
    def test_obvious_non_spam(self):
        # Test if the prediction is 0 on an obvious non-spam input text
        prediction, propensity = score(self.test_non_spam, self.model, self.threshold)
        self.assertEqual(prediction, 0)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


