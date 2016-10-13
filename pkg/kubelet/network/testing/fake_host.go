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

package testing

// helper for testing plugins
// a fake host is created here that can be used by plugins for testing

import (
	"k8s.io/kubernetes/pkg/kubelet/network/hostport"
)

type fakeNetworkHost struct {
	netNS          map[string]string
	annotations    map[string]string
	podPortMapping []*hostport.PodPortMapping
}

func NewFakeHost(netNS map[string]string, annotations map[string]string, podPortMapping []*hostport.PodPortMapping) *fakeNetworkHost {
	return &fakeNetworkHost{
		netNS:          netNS,
		annotations:    annotations,
		podPortMapping: podPortMapping,
	}
}

func (nh *fakeNetworkHost) GetPodSandboxNetNS(podSandboxID string) (string, error) {
	netns, ok := nh.netNS[podSandboxID]
	if !ok {
		return "", nil
	}
	return netns, nil
}

func (nh *fakeNetworkHost) GetPodHostportMapping() ([]*hostport.PodPortMapping, error) {
	return nh.podPortMapping, nil
}

func (nh *fakeNetworkHost) GetPodAnnotations(namespace, name, podSandboxID string) (map[string]string, error) {
	return nh.annotations, nil
}
