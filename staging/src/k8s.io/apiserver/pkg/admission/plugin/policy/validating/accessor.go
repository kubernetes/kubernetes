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

package validating

import (
	"k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
)

func NewValidatingAdmissionPolicyAccessor(obj *v1.ValidatingAdmissionPolicy) generic.PolicyAccessor {
	return &validatingAdmissionPolicyAccessor{
		ValidatingAdmissionPolicy: obj,
	}
}

func NewValidatingAdmissionPolicyBindingAccessor(obj *v1.ValidatingAdmissionPolicyBinding) generic.BindingAccessor {
	return &validatingAdmissionPolicyBindingAccessor{
		ValidatingAdmissionPolicyBinding: obj,
	}
}

type validatingAdmissionPolicyAccessor struct {
	*v1.ValidatingAdmissionPolicy
}

func (v *validatingAdmissionPolicyAccessor) GetNamespace() string {
	return v.Namespace
}

func (v *validatingAdmissionPolicyAccessor) GetName() string {
	return v.Name
}

func (v *validatingAdmissionPolicyAccessor) GetParamKind() *v1.ParamKind {
	return v.Spec.ParamKind
}

func (v *validatingAdmissionPolicyAccessor) GetMatchConstraints() *v1.MatchResources {
	return v.Spec.MatchConstraints
}

func (v *validatingAdmissionPolicyAccessor) GetFailurePolicy() *v1.FailurePolicyType {
	return v.Spec.FailurePolicy
}

type validatingAdmissionPolicyBindingAccessor struct {
	*v1.ValidatingAdmissionPolicyBinding
}

func (v *validatingAdmissionPolicyBindingAccessor) GetNamespace() string {
	return v.Namespace
}

func (v *validatingAdmissionPolicyBindingAccessor) GetName() string {
	return v.Name
}

func (v *validatingAdmissionPolicyBindingAccessor) GetPolicyName() types.NamespacedName {
	return types.NamespacedName{
		Namespace: "",
		Name:      v.Spec.PolicyName,
	}
}

func (v *validatingAdmissionPolicyBindingAccessor) GetMatchResources() *v1.MatchResources {
	return v.Spec.MatchResources
}

func (v *validatingAdmissionPolicyBindingAccessor) GetParamRef() *v1.ParamRef {
	return v.Spec.ParamRef
}
