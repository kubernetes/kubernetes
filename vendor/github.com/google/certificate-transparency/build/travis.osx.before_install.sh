#!/bin/bash

if [ "${TRAVIS_OS_NAME}" != "osx" ]; then
	echo "This script is only useful for OSX on Travis builds!"
	exit 1
fi

export MAKE=gnumake

echo "number of cores:"
sysctl -n hw.ncpu

brew update > /dev/null
brew upgrade go
for i in homebrew/dupes/tcl-tk ant; do
	brew install $i
done
