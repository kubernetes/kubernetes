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

package mutating

import (
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

type key struct {
	PolicyUID     types.NamespacedName
	BindingUID    types.NamespacedName
	ParamUID      types.NamespacedName
	MutationIndex int
}

type policyReinvokeContext struct {
	// lastPolicyOutput holds the result of the last Policy admission plugin call
	lastPolicyOutput runtime.Object
	// previouslyInvokedReinvocablePolicys holds the set of policies that have been invoked and
	// should be reinvoked if a later mutation occurs
	previouslyInvokedReinvocablePolicies sets.Set[key]
	// reinvokePolicies holds the set of Policies that should be reinvoked
	reinvokePolicies sets.Set[key]
}

func (rc *policyReinvokeContext) ShouldReinvoke(policy key) bool {
	return rc.reinvokePolicies.Has(policy)
}

func (rc *policyReinvokeContext) IsOutputChangedSinceLastPolicyInvocation(object runtime.Object) bool {
	return !apiequality.Semantic.DeepEqual(rc.lastPolicyOutput, object)
}

func (rc *policyReinvokeContext) SetLastPolicyInvocationOutput(object runtime.Object) {
	if object == nil {
		rc.lastPolicyOutput = nil
		return
	}
	rc.lastPolicyOutput = object.DeepCopyObject()
}

func (rc *policyReinvokeContext) AddReinvocablePolicyToPreviouslyInvoked(policy key) {
	if rc.previouslyInvokedReinvocablePolicies == nil {
		rc.previouslyInvokedReinvocablePolicies = sets.New[key]()
	}
	rc.previouslyInvokedReinvocablePolicies.Insert(policy)
}

func (rc *policyReinvokeContext) RequireReinvokingPreviouslyInvokedPlugins() {
	if len(rc.previouslyInvokedReinvocablePolicies) > 0 {
		if rc.reinvokePolicies == nil {
			rc.reinvokePolicies = sets.New[key]()
		}
		for s := range rc.previouslyInvokedReinvocablePolicies {
			rc.reinvokePolicies.Insert(s)
		}
		rc.previouslyInvokedReinvocablePolicies = sets.New[key]()
	}
}
