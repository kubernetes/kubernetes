# Copyright 2017 The Kubernetes Authors.
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

.PHONY: all image push clean

TAG = 1.2
PREFIX = gcr.io/google_containers


all: push

nnp: nnp.c
	gcc -static -o $@ $@.c

image: nnp
	docker build --pull -t $(PREFIX)/nonewprivs:$(TAG) .

push: image
	gcloud docker -- push $(PREFIX)/nonewprivs:$(TAG)

clean:
	rm -f nnp
