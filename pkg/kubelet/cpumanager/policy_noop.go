/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
)

type noopPolicy struct{}

var _ Policy = &noopPolicy{}

// PolicyNoop name of noop policy
const PolicyNoop policyName = "noop"

// NewNoopPolicy returns a cupset manager policy that does nothing
func NewNoopPolicy() Policy {
	return &noopPolicy{}
}

func (p *noopPolicy) Name() string {
	return string(PolicyNoop)
}

func (p *noopPolicy) Start(s state.State) {
	glog.Info("[cpumanager] noop policy: Start")
}

func (p *noopPolicy) RegisterContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error {
	glog.Infof("[cpumanager] noop policy: RegisterContainer [%s] (%s) of pod %s", containerID, container.Name, pod.Name)
	return nil
}

func (p *noopPolicy) UnregisterContainer(s state.State, containerID string) error {
	glog.Infof("[cpumanager] noop policy: UnregisterContainer [%s]", containerID)
	return nil
}
