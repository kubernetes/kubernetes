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

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network/hostport"
)

type fakeHandler struct{}

func NewFakeHostportHandler() hostport.HostportHandler {
	return &fakeHandler{}
}

func (h *fakeHandler) OpenPodHostportsAndSync(newPod *hostport.RunningPod, natInterfaceName string, runningPods []*hostport.RunningPod) error {
	return h.SyncHostports(natInterfaceName, append(runningPods, newPod))
}

func (h *fakeHandler) SyncHostports(natInterfaceName string, runningPods []*hostport.RunningPod) error {
	for _, r := range runningPods {
		if r.IP.To4() == nil {
			return fmt.Errorf("Invalid or missing pod %s IP", kubecontainer.GetPodFullName(r.Pod))
		}
	}

	return nil
}
