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

package testing

import (
	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport"
)

type fakeSyncer struct{}

func NewFakeHostportSyncer() hostport.HostportSyncer {
	return &fakeSyncer{}
}

func (h *fakeSyncer) OpenPodHostportsAndSync(newPortMapping *hostport.PodPortMapping, natInterfaceName string, activePortMapping []*hostport.PodPortMapping) error {
	return h.SyncHostports(natInterfaceName, activePortMapping)
}

func (h *fakeSyncer) SyncHostports(natInterfaceName string, activePortMapping []*hostport.PodPortMapping) error {
	for _, r := range activePortMapping {
		if r.IP.To4() == nil {
			return fmt.Errorf("Invalid or missing pod %s/%s IP", r.Namespace, r.Name)
		}
	}

	return nil
}
