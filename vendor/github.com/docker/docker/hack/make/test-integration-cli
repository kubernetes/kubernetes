#!/bin/bash
set -e

bundle_test_integration_cli() {
	TESTFLAGS="$TESTFLAGS -check.v"
	go_test_dir ./integration-cli
}

# subshell so that we can export PATH without breaking other things
(
	bundle .integration-daemon-start

	bundle .integration-daemon-setup

	bundle_test_integration_cli

	bundle .integration-daemon-stop
) 2>&1 | tee -a "$DEST/test.log"
