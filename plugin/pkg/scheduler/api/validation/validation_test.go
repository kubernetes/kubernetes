/*
Copyright 2015 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

func TestValidatePriorityWithNoWeight(t *testing.T) {
	policy := api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority"}}}
	if ValidatePolicy(policy) == nil {
		t.Errorf("Expected error about priority weight not being positive")
	}
}

func TestValidatePriorityWithZeroWeight(t *testing.T) {
	policy := api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}}
	if ValidatePolicy(policy) == nil {
		t.Errorf("Expected error about priority weight not being positive")
	}
}

func TestValidatePriorityWithNonZeroWeight(t *testing.T) {
	policy := api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}}
	errs := ValidatePolicy(policy)
	if errs != nil {
		t.Errorf("Unexpected errors %v", errs)
	}
}

func TestValidatePriorityWithNegativeWeight(t *testing.T) {
	policy := api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}}
	if ValidatePolicy(policy) == nil {
		t.Errorf("Expected error about priority weight not being positive")
	}
}

func TestValidatePriorityWithOverFlowWeight(t *testing.T) {
	policy := api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: api.MaxWeight}}}
	if ValidatePolicy(policy) == nil {
		t.Errorf("Expected error about priority weight not being overflown.")
	}
}

func TestValidateExtenderWithNonNegativeWeight(t *testing.T) {
	extenderPolicy := api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter", Weight: 2}}}
	errs := ValidatePolicy(extenderPolicy)
	if errs != nil {
		t.Errorf("Unexpected errors %v", errs)
	}
}

func TestValidateExtenderWithNegativeWeight(t *testing.T) {
	extenderPolicy := api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter", Weight: -2}}}
	if ValidatePolicy(extenderPolicy) == nil {
		t.Errorf("Expected error about priority weight for extender not being positive")
	}
}

func TestValidateMultipleExtendersWithBind(t *testing.T) {
	extenderPolicy := api.Policy{
		ExtenderConfigs: []api.ExtenderConfig{
			{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind"},
			{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind"},
		},
	}
	if ValidatePolicy(extenderPolicy) == nil {
		t.Errorf("Expected failure when multiple extenders with bind")
	}
}
