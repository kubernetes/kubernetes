import unittest

class BaseCheckTest(unittest.TestCase):
    def assertObservationIn(self, obs, result):
        """Asserts that Observation (or some subtype) with (description, reason,
        details) is in result."""
        self.assertIn((obs.description, obs.reason, obs.details),
                      [(res.description, res.reason, res.details)
                       for res in result])
