/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta1_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	versioned "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// TestJobSpecConversion tests that ManualSelector and AutoSelector
// are handled correctly.
func TestJobSpecConversion(t *testing.T) {
	pTrue := new(bool)
	*pTrue = true
	pFalse := new(bool)
	*pFalse = false

	// False or nil convert to true.
	// True converts to nil.
	tests := []struct {
		in        *bool
		expectOut *bool
	}{
		{
			in:        nil,
			expectOut: pTrue,
		},
		{
			in:        pFalse,
			expectOut: pTrue,
		},
		{
			in:        pTrue,
			expectOut: nil,
		},
	}

	// Test internal -> v1beta1.
	for _, test := range tests {
		i := &batch.JobSpec{
			ManualSelector: test.in,
		}
		v := versioned.JobSpec{}
		if err := api.Scheme.Convert(i, &v); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectOut, v.AutoSelector) {
			t.Fatalf("want v1beta1.AutoSelector %v, got %v", test.expectOut, v.AutoSelector)
		}
	}

	// Test v1beta1 -> internal.
	for _, test := range tests {
		i := &versioned.JobSpec{
			AutoSelector: test.in,
		}
		e := batch.JobSpec{}
		if err := api.Scheme.Convert(i, &e); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectOut, e.ManualSelector) {
			t.Fatalf("want extensions.ManualSelector %v, got %v", test.expectOut, e.ManualSelector)
		}
	}
}
