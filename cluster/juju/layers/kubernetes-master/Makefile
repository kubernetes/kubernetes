#!/usr/bin/make

all: lint unit_test


.PHONY: clean
clean:
	@rm -rf .tox

.PHONY: apt_prereqs
apt_prereqs:
	@# Need tox, but don't install the apt version unless we have to (don't want to conflict with pip)
	@which tox >/dev/null || (sudo pip install tox)
	@sudo apt-get install -y python-pip python-flake8 python3-flake8 flake8

.PHONY: lint
lint: apt_prereqs
	@tox --notest
	@flake8 $(wildcard hooks reactive lib unit_tests tests)
	@charm proof

.PHONY: unit_test
unit_test: apt_prereqs
	@echo Starting tests...
	tox
