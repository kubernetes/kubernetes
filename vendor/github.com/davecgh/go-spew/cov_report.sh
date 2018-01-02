#!/bin/sh

# This script uses gocov to generate a test coverage report.
# The gocov tool my be obtained with the following command:
#   go get github.com/axw/gocov/gocov
#
# It will be installed to $GOPATH/bin, so ensure that location is in your $PATH.

# Check for gocov.
if ! type gocov >/dev/null 2>&1; then
	echo >&2 "This script requires the gocov tool."
	echo >&2 "You may obtain it with the following command:"
	echo >&2 "go get github.com/axw/gocov/gocov"
	exit 1
fi

# Only run the cgo tests if gcc is installed.
if type gcc >/dev/null 2>&1; then
	(cd spew && gocov test -tags testcgo | gocov report)
else
	(cd spew && gocov test | gocov report)
fi
