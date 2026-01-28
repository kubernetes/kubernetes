/*
Copyright The Kubernetes Authors.

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

package resource

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestMaxWithNilResourceList(t *testing.T) {
	tests := []struct {
		name string
		a    corev1.ResourceList
		b    []corev1.ResourceList
		want corev1.ResourceList
	}{
		{
			name: "nil first argument with non-nil second",
			a:    nil,
			b:    []corev1.ResourceList{{corev1.ResourceCPU: resource.MustParse("100m")}},
			want: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		},
		{
			name: "nil first argument with nil second",
			a:    nil,
			b:    []corev1.ResourceList{nil},
			want: corev1.ResourceList{},
		},
		{
			name: "nil first argument with empty second",
			a:    nil,
			b:    []corev1.ResourceList{{}},
			want: corev1.ResourceList{},
		},
		{
			name: "nil first argument with no second arguments",
			a:    nil,
			b:    nil,
			want: corev1.ResourceList{},
		},
		{
			name: "empty first argument with non-nil second",
			a:    corev1.ResourceList{},
			b:    []corev1.ResourceList{{corev1.ResourceCPU: resource.MustParse("100m")}},
			want: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		},
		{
			name: "non-nil first argument takes max",
			a:    corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
			b:    []corev1.ResourceList{{corev1.ResourceCPU: resource.MustParse("100m")}},
			want: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
		},
		{
			name: "second argument larger takes max",
			a:    corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			b:    []corev1.ResourceList{{corev1.ResourceCPU: resource.MustParse("200m")}},
			want: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := max(tt.a, tt.b...)
			if len(got) != len(tt.want) {
				t.Errorf("case %q, expected %d resources but got %d", tt.name, len(tt.want), len(got))
				return
			}
			for name, wantQty := range tt.want {
				gotQty, ok := got[name]
				if !ok {
					t.Errorf("case %q, expected resource %s but it was missing", tt.name, name)
					continue
				}
				if gotQty.Cmp(wantQty) != 0 {
					t.Errorf("case %q, expected resource %s to be %s but got %s", tt.name, name, wantQty.String(), gotQty.String())
				}
			}
		})
	}
}
