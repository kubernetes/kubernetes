// +build !linux

/*
Copyright 2014 The Kubernetes Authors.

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

package kubenet

import (
	"fmt"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
)

type kubenetNetworkPlugin struct {
	network.NoopNetworkPlugin
}

func NewPlugin(networkPluginDir string) network.NetworkPlugin {
	return &kubenetNetworkPlugin{}
}

func (plugin *kubenetNetworkPlugin) Init(host network.Host, hairpinMode componentconfig.HairpinMode, nonMasqueradeCIDR string) error {
	return fmt.Errorf("Kubenet is not supported in this build")
}

func (plugin *kubenetNetworkPlugin) Name() string {
	return "kubenet"
}

func (plugin *kubenetNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID) error {
	return fmt.Errorf("Kubenet is not supported in this build")
}

func (plugin *kubenetNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.ContainerID) error {
	return fmt.Errorf("Kubenet is not supported in this build")
}

func (plugin *kubenetNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	return nil, fmt.Errorf("Kubenet is not supported in this build")
}
