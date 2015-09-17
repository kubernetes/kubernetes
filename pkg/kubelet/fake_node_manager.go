/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net"

	"k8s.io/kubernetes/pkg/api"
)

type fakeNodeManager struct {
	podCIDR string
	node    *api.Node
	IP      net.IP
}

var _ nodeManager = &fakeNodeManager{}

func (f *fakeNodeManager) Start() {
}

func (f *fakeNodeManager) GetNode() (*api.Node, error) {
	return f.node, nil
}

func (f *fakeNodeManager) GetHostIP() (net.IP, error) {
	return f.IP, nil
}

func (f *fakeNodeManager) GetPodCIDR() string {
	return f.podCIDR
}
