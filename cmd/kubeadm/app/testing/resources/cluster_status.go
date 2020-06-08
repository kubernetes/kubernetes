/*
Copyright 2020 The Kubernetes Authors.

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

package resources

import (
	"encoding/json"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// ClusterStatusWithAPIEndpoint returns a FakeConfigMap containing a
// cluster status with the provided endpoint for nodeName as a single
// entry
func ClusterStatusWithAPIEndpoint(nodeName string, endpoint kubeadmapi.APIEndpoint) *FakeConfigMap {
	marshaledClusterStatus, _ := json.Marshal(kubeadmapi.ClusterStatus{
		APIEndpoints: map[string]kubeadmapi.APIEndpoint{
			nodeName: endpoint,
		},
	})
	return &FakeConfigMap{
		Name: constants.KubeadmConfigConfigMap,
		Data: map[string]string{
			constants.ClusterStatusConfigMapKey: string(marshaledClusterStatus),
		},
	}
}
