# Copyright 2018 The Prometheus Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include Makefile.common

%/.unpacked: %.ttar
	@echo ">> extracting fixtures"
	./ttar -C $(dir $*) -x -f $*.ttar
	touch $@

fixtures: fixtures/.unpacked

update_fixtures:
	rm -vf fixtures/.unpacked
	./ttar -c -f fixtures.ttar fixtures/

.PHONY: build
build:

.PHONY: test
test: fixtures/.unpacked common-test
