/*
Copyright 2016 The Kubernetes Authors.

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

package images

import (
	"fmt"
	"runtime"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// GetCoreImage generates and returns the image for the core Kubernetes components or returns overrideImage if specified
func GetCoreImage(image, repoPrefix, k8sVersion, overrideImage string) string {
	if overrideImage != "" {
		return overrideImage
	}
	kubernetesImageTag := kubeadmutil.KubernetesVersionToImageTag(k8sVersion)
	return map[string]string{
		constants.Etcd:                  fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "etcd", runtime.GOARCH, constants.DefaultEtcdVersion),
		constants.KubeAPIServer:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-apiserver", runtime.GOARCH, kubernetesImageTag),
		constants.KubeControllerManager: fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-controller-manager", runtime.GOARCH, kubernetesImageTag),
		constants.KubeScheduler:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-scheduler", runtime.GOARCH, kubernetesImageTag),
	}[image]
}
