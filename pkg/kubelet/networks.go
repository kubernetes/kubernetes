/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// This just exports required functions from kubelet proper, for use by network
// plugins.
type networkHost struct {
	kubelet *Kubelet
}

func (nh *networkHost) GetPodByName(name, namespace string) (*api.Pod, bool) {
	return nh.kubelet.GetPodByName(name, namespace)
}

func (nh *networkHost) GetKubeClient() client.Interface {
	return nh.kubelet.kubeClient
}

func (nh *networkHost) GetRuntime() kubecontainer.Runtime {
	return nh.kubelet.GetRuntime()
}
