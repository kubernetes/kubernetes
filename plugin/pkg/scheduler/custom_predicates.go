/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package scheduler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

func CreateOnMinion1(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	glog.V(2).Infof("custom predicate minion1 --> node: %s", node)
	if node == "10.245.2.2" {
		glog.V(2).Infof("custom predicate minion1 matched")
		return true, nil
	} else {
		glog.V(2).Infof("custom predicate minion1 did not match")
		return false, nil
	}
}

func CreateOnMinion2(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	glog.V(2).Infof("custom predicate minion2 --> node: %s", node)
	if node == "10.245.2.3" {
		glog.V(2).Infof("custom predicate minion2 matched")
		return true, nil
	} else {
		glog.V(2).Infof("custom predicate minion2 did not match")
		return false, nil
	}
}
