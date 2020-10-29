/*
Copyright 2023 The Kubernetes Authors.

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

package apis

import (
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	NodeLabelControlPlane = "node-role.kubernetes.io/control-plane"
	NodeLabelMaster       = "node-role.kubernetes.io/master"
	NodeLabelWorker       = "node-role.kubernetes.io/worker"
	NodeLabelEtcd         = "node-role.kubernetes.io/etcd"
)

var openshiftNodeLabels = sets.NewString(
	NodeLabelControlPlane,
	NodeLabelMaster,
	NodeLabelWorker,
	NodeLabelEtcd,
)

func OpenShiftNodeLabels() []string {
	return openshiftNodeLabels.List()
}

func IsForbiddenOpenshiftLabel(label string) bool {
	return openshiftNodeLabels.Has(label)
}
