/*
Copyright 2023 The Kubernetes Authors.

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

package topologymanager

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type noneScope struct {
	scope
}

// Ensure noneScope implements Scope interface
var _ Scope = &noneScope{}

// NewNoneScope returns a none scope.
func NewNoneScope(recorder record.EventRecorder) Scope {
	return &noneScope{
		scope{
			name:             noneTopologyScope,
			recorder:         recorder,
			podTopologyHints: podTopologyHints{},
			policy:           NewNonePolicy(),
			podMap:           containermap.NewContainerMap(),
		},
	}
}

func (s *noneScope) Admit(pod *v1.Pod) lifecycle.PodAdmitResult {
	return s.admitPolicyNone(pod)
}
