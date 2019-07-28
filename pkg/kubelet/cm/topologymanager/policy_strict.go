/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type strictPolicy struct{}

var _ Policy = &strictPolicy{}

// PolicyStrict policy name.
const PolicyStrict string = "strict"

// NewStrictPolicy returns strict policy.
func NewStrictPolicy() Policy {
	return &strictPolicy{}
}

func (p *strictPolicy) Name() string {
	return PolicyStrict
}

func (p *strictPolicy) CanAdmitPodResult(admit bool) lifecycle.PodAdmitResult {
	if !admit {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  "Topology Affinity Error",
			Message: "Resources cannot be allocated with Topology Locality",
		}
	}
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
