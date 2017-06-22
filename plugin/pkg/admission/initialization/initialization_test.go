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
	"testing"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
