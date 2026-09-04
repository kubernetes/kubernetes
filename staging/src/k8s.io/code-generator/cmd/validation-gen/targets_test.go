/*
Copyright 2026 The Kubernetes Authors.

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

package main

import (
	"maps"
	"testing"
)

func TestMapReadOnlyPkgs(t *testing.T) {
	const (
		metav1     = "k8s.io/apimachinery/pkg/apis/meta/v1"
		metav1val  = "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
		quantity   = "k8s.io/apimachinery/pkg/api/resource"
		generated  = "example.com/api/v1"
		generating = "example.com/apis/v1"
	)

	tests := []struct {
		name          string
		readOnlyPkgs  []string
		readOnlyInput map[string]string
		existing      map[string]string
		want          map[string]string
	}{
		{
			name:          "package with no input tag stands for itself",
			readOnlyPkgs:  []string{quantity},
			readOnlyInput: map[string]string{quantity: quantity},
			want:          map[string]string{quantity: quantity},
		},
		{
			name:          "package with an input tag maps its types package",
			readOnlyPkgs:  []string{metav1val},
			readOnlyInput: map[string]string{metav1val: metav1},
			want:          map[string]string{metav1: metav1val},
		},
		{
			name:          "types package listed before the package holding its validations",
			readOnlyPkgs:  []string{metav1, metav1val},
			readOnlyInput: map[string]string{metav1: metav1, metav1val: metav1},
			want:          map[string]string{metav1: metav1val},
		},
		{
			name:          "types package listed after the package holding its validations",
			readOnlyPkgs:  []string{metav1val, metav1},
			readOnlyInput: map[string]string{metav1: metav1, metav1val: metav1},
			want:          map[string]string{metav1: metav1val},
		},
		{
			name:          "a generation mapping is not overridden",
			readOnlyPkgs:  []string{metav1, metav1val},
			readOnlyInput: map[string]string{metav1: metav1, metav1val: metav1},
			existing:      map[string]string{metav1: generating},
			want:          map[string]string{metav1: generating},
		},
		{
			name:          "unrelated generation mappings are left alone",
			readOnlyPkgs:  []string{quantity},
			readOnlyInput: map[string]string{quantity: quantity},
			existing:      map[string]string{generated: generating},
			want:          map[string]string{generated: generating, quantity: quantity},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := map[string]string{}
			maps.Copy(got, tt.existing)
			mapReadOnlyPkgs(tt.readOnlyPkgs, tt.readOnlyInput, got)
			if !maps.Equal(got, tt.want) {
				t.Errorf("mapReadOnlyPkgs() = %v, want %v", got, tt.want)
			}
		})
	}
}
