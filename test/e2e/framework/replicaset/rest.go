/*
Copyright 2019 The Kubernetes Authors.

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

package replicaset

import (
	appsv1 "k8s.io/api/apps/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

// UpdateReplicaSetWithRetries updates replicaset template with retries.
func UpdateReplicaSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateReplicaSetFunc) (*appsv1.ReplicaSet, error) {
	return testutils.UpdateReplicaSetWithRetries(c, namespace, name, applyUpdate, framework.Logf, framework.Poll, framework.PollShortTimeout)
}
