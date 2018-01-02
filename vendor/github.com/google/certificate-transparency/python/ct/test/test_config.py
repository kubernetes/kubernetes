"""Test meta-data and configuration."""

import os

CRYPTO_TEST_DATA_DIR = "ct/crypto/testdata/"
CERT_ANALYSIS_DATA_DIR = "ct/cert_analysis/test_data"

def get_test_file_path(filename):
    return os.path.join(os.curdir, CRYPTO_TEST_DATA_DIR, filename)

def get_tld_directory():
    return os.path.join(os.curdir, CERT_ANALYSIS_DATA_DIR)
