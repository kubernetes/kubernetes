/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

////////////////////////////////////////////////////////////////////////////////
// Fake Policy Definitions
////////////////////////////////////////////////////////////////////////////////

type FakePolicyDefinition struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Function called when `Matches` is called
	// If nil, a default function that always returns true is used
	// Specified as a function pointer so that this type is still comparable
	MatchFunc *func(admission.Attributes) bool `json:"-"`

	// Func invoked for implementation of `Compile`
	// Specified as a function pointer so that this type is still comparable
	CompileFunc *func(converter ObjectConverter) (EvaluatorFunc, error) `json:"-"`

	// GVK to return when ParamSource() is called
	ParamSource *schema.GroupVersionKind `json:"paramSource"`

	FailurePolicy FailurePolicy `json:"failurePolicy"`
}

var _ PolicyDefinition = &FakePolicyDefinition{}

func (f *FakePolicyDefinition) SetGroupVersionKind(kind schema.GroupVersionKind) {
	f.TypeMeta.APIVersion = kind.GroupVersion().String()
	f.TypeMeta.Kind = kind.Kind
}

func (f *FakePolicyDefinition) GroupVersionKind() schema.GroupVersionKind {
	parsedGV, err := schema.ParseGroupVersion(f.TypeMeta.APIVersion)
	if err != nil || f.TypeMeta.Kind == "" || parsedGV.Empty() {
		return schema.GroupVersionKind{
			Group:   "admission.k8s.io",
			Version: "v1alpha1",
			Kind:    "PolicyDefinition",
		}
	}
	return schema.GroupVersionKind{
		Group:   parsedGV.Group,
		Version: parsedGV.Version,
		Kind:    f.TypeMeta.Kind,
	}
}

func (f *FakePolicyDefinition) GetObjectKind() schema.ObjectKind {
	return f
}

func (f *FakePolicyDefinition) DeepCopyObject() runtime.Object {
	copied := *f
	f.ObjectMeta.DeepCopyInto(&copied.ObjectMeta)
	return &copied
}

func (f *FakePolicyDefinition) GetName() string {
	return f.ObjectMeta.Name
}

func (f *FakePolicyDefinition) GetNamespace() string {
	return f.ObjectMeta.Namespace
}

func (f *FakePolicyDefinition) Matches(a admission.Attributes) bool {
	if f.MatchFunc == nil || *f.MatchFunc == nil {
		return true
	}
	return (*f.MatchFunc)(a)
}

func (f *FakePolicyDefinition) Compile(
	converter ObjectConverter,
	mapper meta.RESTMapper,
) (EvaluatorFunc, error) {
	if f.CompileFunc == nil || *f.CompileFunc == nil {
		panic("must provide a CompileFunc to policy definition")
	}
	return (*f.CompileFunc)(converter)
}

func (f *FakePolicyDefinition) GetParamSource() *schema.GroupVersionKind {
	return f.ParamSource
}

func (f *FakePolicyDefinition) GetFailurePolicy() FailurePolicy {
	return f.FailurePolicy
}

////////////////////////////////////////////////////////////////////////////////
// Fake Policy Binding
////////////////////////////////////////////////////////////////////////////////

type FakePolicyBinding struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Specified as a function pointer so that this type is still comparable
	MatchFunc *func(admission.Attributes) bool `json:"-"`
	Params    string                           `json:"params"`
	Policy    string                           `json:"policy"`
}

var _ PolicyBinding = &FakePolicyBinding{}

func (f *FakePolicyBinding) SetGroupVersionKind(kind schema.GroupVersionKind) {
	f.TypeMeta.APIVersion = kind.GroupVersion().String()
	f.TypeMeta.Kind = kind.Kind
}

func (f *FakePolicyBinding) GroupVersionKind() schema.GroupVersionKind {
	parsedGV, err := schema.ParseGroupVersion(f.TypeMeta.APIVersion)
	if err != nil || f.TypeMeta.Kind == "" || parsedGV.Empty() {
		return schema.GroupVersionKind{
			Group:   "admission.k8s.io",
			Version: "v1alpha1",
			Kind:    "PolicyBinding",
		}
	}
	return schema.GroupVersionKind{
		Group:   parsedGV.Group,
		Version: parsedGV.Version,
		Kind:    f.TypeMeta.Kind,
	}
}

func (f *FakePolicyBinding) GetObjectKind() schema.ObjectKind {
	return f
}

func (f *FakePolicyBinding) DeepCopyObject() runtime.Object {
	copied := *f
	f.ObjectMeta.DeepCopyInto(&copied.ObjectMeta)
	return &copied
}

func (f *FakePolicyBinding) Matches(a admission.Attributes) bool {
	if f.MatchFunc == nil || *f.MatchFunc == nil {
		return true
	}
	return (*f.MatchFunc)(a)
}

func (f *FakePolicyBinding) GetTargetDefinition() (namespace, name string) {
	return f.Namespace, f.Policy
}

func (f *FakePolicyBinding) GetTargetParams() (namespace, name string) {
	return f.Namespace, f.Params
}

/// List Types

type FakePolicyDefinitionList struct {
	metav1.TypeMeta
	metav1.ListMeta

	Items []FakePolicyDefinition
}

func (f *FakePolicyDefinitionList) SetGroupVersionKind(kind schema.GroupVersionKind) {
	f.TypeMeta.APIVersion = kind.GroupVersion().String()
	f.TypeMeta.Kind = kind.Kind
}

func (f *FakePolicyDefinitionList) GroupVersionKind() schema.GroupVersionKind {
	parsedGV, err := schema.ParseGroupVersion(f.TypeMeta.APIVersion)
	if err != nil || f.TypeMeta.Kind == "" || parsedGV.Empty() {
		return schema.GroupVersionKind{
			Group:   "admission.k8s.io",
			Version: "v1alpha1",
			Kind:    "PolicyDefinitionList",
		}
	}
	return schema.GroupVersionKind{
		Group:   parsedGV.Group,
		Version: parsedGV.Version,
		Kind:    f.TypeMeta.Kind,
	}
}

func (f *FakePolicyDefinitionList) GetObjectKind() schema.ObjectKind {
	return f
}

func (f *FakePolicyDefinitionList) DeepCopyObject() runtime.Object {
	copied := *f
	f.ListMeta.DeepCopyInto(&copied.ListMeta)
	copied.Items = make([]FakePolicyDefinition, len(f.Items))
	copy(copied.Items, f.Items)
	return &copied
}

type FakePolicyBindingList struct {
	metav1.TypeMeta
	metav1.ListMeta

	Items []FakePolicyBinding
}

func (f *FakePolicyBindingList) SetGroupVersionKind(kind schema.GroupVersionKind) {
	f.TypeMeta.APIVersion = kind.GroupVersion().String()
	f.TypeMeta.Kind = kind.Kind
}

func (f *FakePolicyBindingList) GroupVersionKind() schema.GroupVersionKind {
	parsedGV, err := schema.ParseGroupVersion(f.TypeMeta.APIVersion)
	if err != nil || f.TypeMeta.Kind == "" || parsedGV.Empty() {
		return schema.GroupVersionKind{
			Group:   "admission.k8s.io",
			Version: "v1alpha1",
			Kind:    "PolicyBindingList",
		}
	}
	return schema.GroupVersionKind{
		Group:   parsedGV.Group,
		Version: parsedGV.Version,
		Kind:    f.TypeMeta.Kind,
	}
}

func (f *FakePolicyBindingList) GetObjectKind() schema.ObjectKind {
	return f
}

func (f *FakePolicyBindingList) DeepCopyObject() runtime.Object {
	copied := *f
	f.ListMeta.DeepCopyInto(&copied.ListMeta)
	copied.Items = make([]FakePolicyBinding, len(f.Items))
	copy(copied.Items, f.Items)
	return &copied
}
