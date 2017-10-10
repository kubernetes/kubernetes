/*
Copyright 2017 The Kubernetes Authors.

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

package initialization

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func newInitializer(name string, rules ...v1alpha1.Rule) *v1alpha1.InitializerConfiguration {
	return addInitializer(&v1alpha1.InitializerConfiguration{}, name, rules...)
}

func addInitializer(base *v1alpha1.InitializerConfiguration, name string, rules ...v1alpha1.Rule) *v1alpha1.InitializerConfiguration {
	base.Initializers = append(base.Initializers, v1alpha1.Initializer{
		Name:  name,
		Rules: rules,
	})
	return base
}

func TestFindInitializers(t *testing.T) {
	type args struct {
		initializers *v1alpha1.InitializerConfiguration
		gvr          schema.GroupVersionResource
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			name: "empty",
			args: args{
				gvr:          schema.GroupVersionResource{},
				initializers: newInitializer("1"),
			},
		},
		{
			name: "everything",
			args: args{
				gvr:          schema.GroupVersionResource{},
				initializers: newInitializer("1", v1alpha1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}}),
			},
			want: []string{"1"},
		},
		{
			name: "empty group",
			args: args{
				gvr:          schema.GroupVersionResource{},
				initializers: newInitializer("1", v1alpha1.Rule{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"*"}}),
			},
			want: []string{"1"},
		},
		{
			name: "pod",
			args: args{
				gvr: schema.GroupVersionResource{Resource: "pods"},
				initializers: addInitializer(
					newInitializer("1", v1alpha1.Rule{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"pods"}}),
					"2", v1alpha1.Rule{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"pods"}},
				),
			},
			want: []string{"1", "2"},
		},
		{
			name: "multiple matches",
			args: args{
				gvr:          schema.GroupVersionResource{Resource: "pods"},
				initializers: newInitializer("1", v1alpha1.Rule{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"pods"}}),
			},
			want: []string{"1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findInitializers(tt.args.initializers, tt.args.gvr); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("findInitializers() = %v, want %v", got, tt.want)
			}
		})
	}
}

type fakeAuthorizer struct {
	accept bool
}

func (f *fakeAuthorizer) Authorize(a authorizer.Attributes) (bool, string, error) {
	if f.accept {
		return true, "", nil
	}
	return false, "denied", nil
}

func TestAdmitUpdate(t *testing.T) {
	tests := []struct {
		name             string
		oldInitializers  *metav1.Initializers
		newInitializers  *metav1.Initializers
		verifyUpdatedObj func(runtime.Object) (pass bool, reason string)
		err              string
	}{
		{
			name:            "updates on initialized resources are allowed",
			oldInitializers: nil,
			newInitializers: nil,
			err:             "",
		},
		{
			name:            "updates on initialized resources are allowed",
			oldInitializers: &metav1.Initializers{Pending: []metav1.Initializer{{Name: "init.k8s.io"}}},
			newInitializers: &metav1.Initializers{},
			verifyUpdatedObj: func(obj runtime.Object) (bool, string) {
				accessor, err := meta.Accessor(obj)
				if err != nil {
					return false, "cannot get accessor"
				}
				if accessor.GetInitializers() != nil {
					return false, "expect nil initializers"
				}
				return true, ""
			},
			err: "",
		},
		{
			name:            "initializers may not be set once initialized",
			oldInitializers: nil,
			newInitializers: &metav1.Initializers{Pending: []metav1.Initializer{{Name: "init.k8s.io"}}},
			err:             "field is immutable once initialization has completed",
		},
		{
			name:            "empty initializer list is treated as nil initializer",
			oldInitializers: nil,
			newInitializers: &metav1.Initializers{},
			verifyUpdatedObj: func(obj runtime.Object) (bool, string) {
				accessor, err := meta.Accessor(obj)
				if err != nil {
					return false, "cannot get accessor"
				}
				if accessor.GetInitializers() != nil {
					return false, "expect nil initializers"
				}
				return true, ""
			},
			err: "",
		},
	}

	plugin := initializer{
		config:     nil,
		authorizer: &fakeAuthorizer{true},
	}
	for _, tc := range tests {
		oldObj := &v1.Pod{}
		oldObj.Initializers = tc.oldInitializers
		newObj := &v1.Pod{}
		newObj.Initializers = tc.newInitializers
		a := admission.NewAttributesRecord(newObj, oldObj, schema.GroupVersionKind{}, "", "foo", schema.GroupVersionResource{}, "", admission.Update, nil)
		err := plugin.Admit(a)
		switch {
		case tc.err == "" && err != nil:
			t.Errorf("%q: unexpected error: %v", tc.name, err)
		case tc.err != "" && err == nil:
			t.Errorf("%q: unexpected no error, expected %s", tc.name, tc.err)
		case tc.err != "" && err != nil && !strings.Contains(err.Error(), tc.err):
			t.Errorf("%q: expected %s, got %v", tc.name, tc.err, err)
		}
	}

}
