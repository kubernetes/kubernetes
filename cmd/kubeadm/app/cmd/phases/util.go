/*
Copyright 2018 The Kubernetes Authors.

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

package phases

import (
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/pkg/version"
)

// SetKubernetesVersion gets the current Kubeadm version and sets it as KubeadmVersion in the config,
// unless it's already set to a value different from the default.
func SetKubernetesVersion(cfg *kubeadmapiv1beta1.ClusterConfiguration) {

	if cfg.KubernetesVersion != kubeadmapiv1beta1.DefaultKubernetesVersion && cfg.KubernetesVersion != "" {
		return
	}
	cfg.KubernetesVersion = version.Get().String()
}
