import gflags
import logging

from collections import defaultdict
from ct.cert_analysis import all_checks
from ct.client import reporter

FLAGS = gflags.FLAGS

class TextCertificateReport(reporter.CertificateReport):
    """Stores description of new entries between last verified STH and
    current."""

    def __init__(self, checks=all_checks.ALL_CHECKS):
        super(TextCertificateReport, self).__init__(checks=checks)

    def report(self):
        """Report stored changes and reset report."""
        super(TextCertificateReport, self).report()
        logging.info("Report:")
        logging.info("New entries since last verified STH: %s" %
                     self.new_entries_count)
        logging.info("Number of entries with observations: %d" %
                     self.entries_with_issues)
        logging.info("Observations:")
        # if number of new entries is unknown then we just count percentages
        # based on number of certificates with observations
        logging.info("Stats:")
        for description_reason, count in self.stats.iteritems():
            description, reason = description_reason
            logging.info("%s %s: %d (%.5f%%)"
                         % (description,
                            "(%s)" % reason if reason else '',
                            count,
                            float(count) / self.new_entries_count * 100.))
        self.reset()

    def reset(self):
        """Clean up report.

        It's also ran at start."""
        self.stats = defaultdict(int)
        self.entries_with_issues = 0
        self.new_entries_count = 0

    def _batch_scanned_callback(self, result):
        for _, log_index, cert_observations in result:
            msg = "Cert %d:" % log_index
            observations = [unicode(obs) for obs in cert_observations]
            if observations:
                logging.info("%s %s", msg, ', '.join(observations))
            unique_observations = set((obs.description, obs.reason)
                                      for obs in cert_observations)
            for obs in unique_observations:
                self.stats[obs] += 1
            if len(cert_observations) > 0:
                self.entries_with_issues += 1
            self.new_entries_count += 1
