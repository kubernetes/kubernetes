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

package dockershim

import "k8s.io/kubernetes/pkg/kubelet/network/hostport"

// Exports required functions from dockershim for use by network plugins
type DockerNetworkHost struct {
	ds *dockerService
}

func (nh *DockerNetworkHost) GetPodSandboxNetNS(podSandboxID string) (string, error) {
	return nh.ds.GetPodSandboxNetNS(podSandboxID)
}

func (nh *DockerNetworkHost) GetPodHostportMapping() ([]*hostport.PodPortMapping, error) {
	return nh.ds.GetPodHostportMapping()
}

func (nh *DockerNetworkHost) GetPodAnnotations(namespace, name, podSandboxID string) (map[string]string, error) {
	return nh.ds.GetPodAnnotations(namespace, name, podSandboxID)
}
