#!/usr/bin/env python
"""Generates a list of hashes of EV certificates found in a log."""

import json
import os
import requests
import shutil
import StringIO
import sys
import tempfile
import urlparse
import zipfile

import generate_ev_whitelist
import golomb_code

import gflags

CRX_URL = (
    "https://clients2.googleusercontent.com/crx/blobs/QgAAAC6zw0qH2DJtnXe8Z7rUJP35Av6_uD2R0shIWqC2XwPmpwrDdH5TKyq_6MWRGfvCXV3f-8g220oUjAO-JvMYG2m5CSvsx6cHch05zQS7ZbtrAMZSmuWMzK-Yj7NpgCxtqGhyTFveAvi-IQ/oafdbfcohdcjandcenmccfopbeklnicp_main.platform.all.crx")

TRUNCATED_HASH_LENGTH = 8
GOLOMB_M_PARAMETER = 2 ** 47
NUM_FETCHING_PROCESSES = 40
# Map between a log's URL and the index of the cut-off point after which
# no new EV certificates will be whitelisted.
# These entries are basically the index of the last EV cert that was found on
# January 1st, when the EV whitelist was finalized.
# Ultimately this allows re-generation of the whitelist, excluding certificates
# that have recently expired, thus reducing its size.
LOGS_LIST = {
    "https://ct.googleapis.com/pilot":6082932,
    "https://ct.googleapis.com/aviator":5360225,
    "https://ct1.digicert-ct.com/log":1056,
    "https://log.certly.io": 4729,
    "https://ct.izenpe.com": 1,
}

FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    "output_crx", "chrome_ev_whitelist_%d",
    "Output Chrome extension. "
    "This zip file should be uploaded to the Chrome web store.")

gflags.DEFINE_string("output_truncated_hashes", "raw_truncated_hashes",
                     "The truncated hashes, hex-encoded, before compression.")


class EVCRXHandler(object):
    """Produces updated EV whitelist Chrome extension."""
    def __init__(self, crx_url=CRX_URL):
        req = requests.get(crx_url)
        if not req.ok:
            raise Exception("Error while fetching %s: %d" %
                            crx_url, req.status_code)
        # Skip the CRX header.
        crx_zip_bytes = req.content[566:]
        sio = StringIO.StringIO(crx_zip_bytes)
        zipped_crx = zipfile.ZipFile(sio)
        unpacked_crx_dir = tempfile.mkdtemp(prefix="tmpcrx")
        zipped_crx.extractall(unpacked_crx_dir)
        self._crx_dir = unpacked_crx_dir

    def _read_manifest_json(self):
        """Reads and parses the extension's metadata from manifest.json"""
        with open(os.path.join(self._crx_dir, "manifest.json")) as manifest:
            return json.load(manifest)

    def _write_manifest_json(self, json_to_write):
        """Overwrites the extension's metadata with the provided json."""
        with open(os.path.join(self._crx_dir, "manifest.json"), "wb") as manifest:
            json.dump(json_to_write, manifest)

    def manifest_version(self):
        """Returns the version number from the extension's metadata."""
        manifest = self._read_manifest_json()
        return int(manifest["version"])

    def update_ev_whitelist(self, compressed_ev_whitelist):
        """Writes an updated EV whitelist into the extension.

        Each new EV whitelist must be uploaded with a version number greater
        than the current version, so increments the version number.
        """
        manifest = self._read_manifest_json()
        manifest["version"] = str(self.manifest_version() + 1)
        self._write_manifest_json(manifest)
        with open(
            os.path.join(
                self._crx_dir,
                "_platform_specific",
                "all",
                "ev_hashes_whitelist.bin"),
            "wb") as hashes_file:
            hashes_file.write(compressed_ev_whitelist)

    def pack_to(self, dest_file_template):
        """Packs the extesion to the provided destination.

        Note that the provided dest_file_template is a template string so
        the version number can be filled in.
        Returns the exact file name that was written.
        """
        fname = dest_file_template % self.manifest_version()
        shutil.make_archive(fname, "zip", self._crx_dir)
        return fname

    def cleanup(self):
        """Deletes the directory containing the unpacked crx."""
        shutil.rmtree(self._crx_dir)
        self._crx_dir = None


def generate_new_ev_hashes_whitelist(logs_list):
    """Scans the provided logs_list and generates a compressed EV certs
    whitelist for all EV certs in all the logs."""
    all_hashes_set = set()
    for log_url, last_acceptable_entry in logs_list.items():
        output_dir = None
        if FLAGS.output_directory:
            parsed_url = urlparse.urlsplit(log_url)
            output_dir = os.path.join(
                FLAGS.output_directory,
                parsed_url.netloc,
                parsed_url.path.strip("/"))
        print "Scanning log %s, certificates will be in %s" % (
            log_url, output_dir)
        res, hashes_set = (
            generate_ev_whitelist.generate_ev_cert_hashes_from_log(
                log_url,
                FLAGS.multi,
                output_dir,
                last_acceptable_entry))
        print "Scanned %d, %d matched, %d failed parsing (strict/partial)" % (
            res.total, res.matches, res.errors)
        print "There are %d EV hashes." % (len(hashes_set))
        old_hashes_count = len(all_hashes_set)
        all_hashes_set = all_hashes_set.union(hashes_set)
        print "%d new EV certificates." % (
            (len(all_hashes_set) - old_hashes_count))

    hashes_list = list(all_hashes_set)
    hashes_list.sort()
    with open(FLAGS.output_truncated_hashes, "wb") as hashes_file:
        for trimmed_hash in hashes_list:
            hashes_file.write(trimmed_hash)
    golomb_coded_bytes = golomb_code.golomb_encode(
        hashes_list, FLAGS.hash_length, 2 ** FLAGS.two_power)
    return golomb_coded_bytes


def main():
    crx = EVCRXHandler()
    print "Current manifest version is %d." % (crx.manifest_version())
    compressed_hashes = generate_new_ev_hashes_whitelist(LOGS_LIST)
    crx.update_ev_whitelist(compressed_hashes)
    output_file = crx.pack_to(FLAGS.output_crx)
    print "New EV whitelist is at %s" % (output_file)
    crx.cleanup()

if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    main()
