/*
Copyright 2018 The Kubernetes Authors.

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

package policy

import (
	"k8s.io/api/auditregistration/v1alpha1"
	"k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// ConvertDynamicPolicyToInternal constructs an internal policy type from a
// v1alpha1 dynamic type
func ConvertDynamicPolicyToInternal(p *v1alpha1.Policy) *audit.Policy {
	stages := make([]audit.Stage, len(p.Stages))
	for i, stage := range p.Stages {
		stages[i] = audit.Stage(stage)
	}
	return &audit.Policy{
		Rules: []audit.PolicyRule{
			{
				Level: audit.Level(p.Level),
			},
		},
		OmitStages: InvertStages(stages),
	}
}

// NewDynamicChecker returns a new dynamic policy checker
func NewDynamicChecker() Checker {
	return &dynamicPolicyChecker{}
}

type dynamicPolicyChecker struct{}

// LevelAndStages returns returns a fixed level of the full event, this is so that the downstream policy
// can be applied per sink.
// TODO: this needs benchmarking before the API moves to beta to determine the effect this has on the apiserver
func (d *dynamicPolicyChecker) LevelAndStages(authorizer.Attributes) (audit.Level, []audit.Stage) {
	return audit.LevelRequestResponse, []audit.Stage{}
}
