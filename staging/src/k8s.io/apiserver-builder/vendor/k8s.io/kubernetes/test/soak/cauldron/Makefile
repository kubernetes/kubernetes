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

.PHONY: cauldron

TAG = 1.0

cauldron:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' cauldron.go

container:	cauldron
	docker build --pull -t kubernetes/cauldron:$(TAG) .

push:
	docker push kubernetes/cauldron:$(TAG)

rc:
	kubectl create --validate -f cauldron-rc.yaml

stop:
	kubectl delete rc cauldron

get:
	kubectl get rc,pods -l app=cauldron

scale20:
	kubectl scale rc cauldron --replicas=20

scale1:
	kubectl scale rc cauldron --replicas=1

clean:
	rm -rf cauldron
