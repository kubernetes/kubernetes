# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

build: virtualenv lint test

virtualenv:
	virtualenv .venv
	.venv/bin/pip install -q -r requirements.txt

lint: virtualenv
	@.venv/bin/flake8 hooks --exclude=charmhelpers --ignore=W391
	@.venv/bin/charm proof

test: virtualenv 
	@CHARM_DIR=. PYTHONPATH=./hooks .venv/bin/py.test unit_tests/*

functional-test:
	@bundletester

release: check-path virtualenv
	@.venv/bin/pip install git-vendor
	@.venv/bin/git-vendor sync -d ${KUBERNETES_BZR}

check-path:
ifndef KUBERNETES_BZR
	$(error KUBERNETES_BZR is undefined)
endif

clean:
	rm -rf .venv
	find -name *.pyc -delete
	rm -rf unit_tests/.cache
