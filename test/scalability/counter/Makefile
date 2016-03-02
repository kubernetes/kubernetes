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

counter:
	kubectl create --validate -f counter-rc.yaml

counter1:
	kubectl scale rc counter --replicas=1

counter10:
	kubectl scale rc counter --replicas=10

counter50:
	kubectl scale rc counter --replicas=50

counter100:
	kubectl scale rc counter --replicas=100

counter200:
	kubectl scale rc counter --replicas=200

counter500:
	kubectl scale rc counter --replicas=500


counter5000:
	kubectl scale rc counter --replicas=5000

stop:
	kubectl delete rc counter
