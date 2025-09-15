/*
Copyright 2024 The Kubernetes Authors.

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
	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
)

func NewMutatingAdmissionPolicyAccessor(obj *Policy) generic.PolicyAccessor {
	return &mutatingAdmissionPolicyAccessor{
		Policy: obj,
	}
}

func NewMutatingAdmissionPolicyBindingAccessor(obj *PolicyBinding) generic.BindingAccessor {
	return &mutatingAdmissionPolicyBindingAccessor{
		PolicyBinding: obj,
	}
}

type mutatingAdmissionPolicyAccessor struct {
	*Policy
}

func (v *mutatingAdmissionPolicyAccessor) GetNamespace() string {
	return v.Namespace
}

func (v *mutatingAdmissionPolicyAccessor) GetName() string {
	return v.Name
}

func (v *mutatingAdmissionPolicyAccessor) GetParamKind() *v1.ParamKind {
	pk := v.Spec.ParamKind
	if pk == nil {
		return nil
	}
	return &v1.ParamKind{
		APIVersion: pk.APIVersion,
		Kind:       pk.Kind,
	}
}

func (v *mutatingAdmissionPolicyAccessor) GetMatchConstraints() *v1.MatchResources {
	return convertV1alpha1ResourceRulesToV1(v.Spec.MatchConstraints)
}

func (v *mutatingAdmissionPolicyAccessor) GetFailurePolicy() *v1.FailurePolicyType {
	return toV1FailurePolicy(v.Spec.FailurePolicy)
}

func toV1FailurePolicy(failurePolicy *v1beta1.FailurePolicyType) *v1.FailurePolicyType {
	if failurePolicy == nil {
		return nil
	}
	fp := v1.FailurePolicyType(*failurePolicy)
	return &fp
}

type mutatingAdmissionPolicyBindingAccessor struct {
	*PolicyBinding
}

func (v *mutatingAdmissionPolicyBindingAccessor) GetNamespace() string {
	return v.Namespace
}

func (v *mutatingAdmissionPolicyBindingAccessor) GetName() string {
	return v.Name
}

func (v *mutatingAdmissionPolicyBindingAccessor) GetPolicyName() types.NamespacedName {
	return types.NamespacedName{
		Namespace: "",
		Name:      v.Spec.PolicyName,
	}
}

func (v *mutatingAdmissionPolicyBindingAccessor) GetMatchResources() *v1.MatchResources {
	return convertV1alpha1ResourceRulesToV1(v.Spec.MatchResources)
}

func (v *mutatingAdmissionPolicyBindingAccessor) GetParamRef() *v1.ParamRef {
	if v.Spec.ParamRef == nil {
		return nil
	}

	var nfa *v1.ParameterNotFoundActionType
	if v.Spec.ParamRef.ParameterNotFoundAction != nil {
		nfa = new(v1.ParameterNotFoundActionType)
		*nfa = v1.ParameterNotFoundActionType(*v.Spec.ParamRef.ParameterNotFoundAction)
	}

	return &v1.ParamRef{
		Name:                    v.Spec.ParamRef.Name,
		Namespace:               v.Spec.ParamRef.Namespace,
		Selector:                v.Spec.ParamRef.Selector,
		ParameterNotFoundAction: nfa,
	}
}

func convertV1alpha1ResourceRulesToV1(mc *v1beta1.MatchResources) *v1.MatchResources {
	if mc == nil {
		return nil
	}

	var res v1.MatchResources
	res.NamespaceSelector = mc.NamespaceSelector
	res.ObjectSelector = mc.ObjectSelector
	for _, ex := range mc.ExcludeResourceRules {
		res.ExcludeResourceRules = append(res.ExcludeResourceRules, v1.NamedRuleWithOperations{
			ResourceNames:      ex.ResourceNames,
			RuleWithOperations: ex.RuleWithOperations,
		})
	}
	for _, ex := range mc.ResourceRules {
		res.ResourceRules = append(res.ResourceRules, v1.NamedRuleWithOperations{
			ResourceNames:      ex.ResourceNames,
			RuleWithOperations: ex.RuleWithOperations,
		})
	}
	if mc.MatchPolicy != nil {
		mp := v1.MatchPolicyType(*mc.MatchPolicy)
		res.MatchPolicy = &mp
	}
	return &res
}
