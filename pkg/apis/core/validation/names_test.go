/*
Copyright 2025 The Kubernetes Authors.

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

import "testing"

func TestIsKubernetesSignerName(t *testing.T) {
	testCases := []struct {
		name string
		want bool
	}{
		{
			name: "kubernetes.io",
			want: true,
		},
		{
			name: "kubernetes.io/a",
			want: true,
		},
		{
			name: "kubernetes.io/a/b.c/d.e",
			want: true,
		},
		{
			name: "foo.kubernetes.io",
			want: true,
		},
		{
			name: "fookubernetes.io",
			want: false,
		},
		{
			name: "foo.com/a",
			want: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsKubernetesSignerName(tc.name)
			if got != tc.want {
				t.Errorf("IsKubernetesSignerName(%q); got %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}
