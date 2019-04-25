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
	./ttar -C $(dir $*) -x -f $*.ttar
	touch $@

update_fixtures: fixtures.ttar sysfs/fixtures.ttar

%fixtures.ttar: %/fixtures
	rm -v $(dir $*)fixtures/.unpacked
	./ttar -C $(dir $*) -c -f $*fixtures.ttar fixtures/

.PHONY: build
build:

.PHONY: test
test: fixtures/.unpacked sysfs/fixtures/.unpacked common-test
