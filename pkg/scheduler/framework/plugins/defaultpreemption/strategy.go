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

package defaultpreemption

import (
	v1 "k8s.io/api/core/v1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
)

// Strategy defines a number of operations to share a preemption strategy.
// It's executable by preempting the victims, set the nominatedName for the preemptor,
// as well as other operations to maintain the nominated Pods properly.
type Strategy interface {
	// Victims wraps a list of to-be-preempted Pods and the number of PDB violation.
	Victims() *extenderv1.Victims
	// NominatedNodeName returns the target node name where the preemptor gets nominated to run.
	NominatedNodeName() string
	// NominatedPodsToClear returns the Pods whose nominatedNodeName value should be cleared.
	NominatedPodsToClear() []*v1.Pod
}

type strategy struct {
	victims              *extenderv1.Victims
	nominatedNodeName    string
	nominatedPodsToClear []*v1.Pod
}

// Victims returns s.victims.
func (s *strategy) Victims() *extenderv1.Victims {
	return s.victims
}

// NominatedNodeName returns s.nominatedNodeName.
func (s *strategy) NominatedNodeName() string {
	return s.nominatedNodeName
}

// NominatedPodsToClear returns s.nominatedPodsToClear
func (s *strategy) NominatedPodsToClear() []*v1.Pod {
	return s.nominatedPodsToClear
}
