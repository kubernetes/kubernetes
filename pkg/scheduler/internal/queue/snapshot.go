/*
Copyright 2020 The Kubernetes Authors.

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

package queue

import (
	v1 "k8s.io/api/core/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
)

// NomPodsSnapshot is a structure enables nominated Pods can be fast accessed.
type NomPodsSnapshot struct {
	nominatedPods      map[string][]*v1.Pod
	nominatedPodToNode map[ktypes.UID]string
}

// NewEmptyNomPodsSnapshot instantiates a NomPodsSnapshot.
func NewEmptyNomPodsSnapshot() *NomPodsSnapshot {
	return &NomPodsSnapshot{
		nominatedPods:      make(map[string][]*v1.Pod),
		nominatedPodToNode: make(map[ktypes.UID]string),
	}
}

// PodsForNode returns nominated Pods that are associated with given <nodeName>.
func (s *NomPodsSnapshot) PodsForNode(nodeName string) []*v1.Pod {
	if list, ok := s.nominatedPods[nodeName]; ok {
		return list
	}
	return nil
}
