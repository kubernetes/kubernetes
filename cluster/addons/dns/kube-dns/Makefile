# Copyright 2016 The Kubernetes Authors.
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

# Makefile for the kubedns underscore templates to Salt/Pillar and other formats.

# If you update the *.base templates, please run this Makefile before pushing.
#
# Usage:
#    make

all: transform

# .base -> .in pattern rule
%.in: %.base
	sed -f transforms2salt.sed $< | sed s/__SOURCE_FILENAME__/$</g > $@

# .base -> .sed pattern rule
%.sed: %.base
	sed -f transforms2sed.sed $<  | sed s/__SOURCE_FILENAME__/$</g > $@

transform: kube-dns.yaml.in kube-dns.yaml.sed

.PHONY: transform
