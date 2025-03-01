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

package protobuf

import (
	"bytes"
	"encoding/base64"
	"github.com/google/go-cmp/cmp"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestCollectionsEncoding(t *testing.T) {
	t.Run("Normal", func(t *testing.T) {
		testCollectionsEncoding(t, NewSerializer(nil, nil))
	})
	// Leave place for testing streaming collection serializer proposed as part of KEP-5116
}

func testCollectionsEncoding(t *testing.T, s *Serializer) {
	var remainingItems int64 = 1
	testCases := []struct {
		name string
		in   runtime.Object
		// expect is base64 encoded protobuf bytes
		expect string
	}{
		{
			name: "CarpList items nil",
			in: &testapigroupv1.CarpList{
				Items: nil,
			},
			expect: "azhzAAoECgASABIICgYKABIAGgAaACIA",
		},
		{
			name: "CarpList slice nil",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Status: testapigroupv1.CarpStatus{
							Conditions: nil,
						},
					},
				},
			},
			expect: "azhzAAoECgASABJBCgYKABIAGgASNwoQCgASABoAIgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgAaACIA",
		},
		{
			name: "CarpList map nil",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Spec: testapigroupv1.CarpSpec{
							NodeSelector: nil,
						},
					},
				},
			},
			expect: "azhzAAoECgASABJBCgYKABIAGgASNwoQCgASABoAIgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgAaACIA",
		},
		{
			name: "CarpList items empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{},
			},
			expect: "azhzAAoECgASABIICgYKABIAGgAaACIA",
		},
		{
			name: "CarpList slice empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Status: testapigroupv1.CarpStatus{
							Conditions: []testapigroupv1.CarpCondition{},
						},
					},
				},
			},
			expect: "azhzAAoECgASABJBCgYKABIAGgASNwoQCgASABoAIgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgAaACIA",
		},
		{
			name: "CarpList map empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Spec: testapigroupv1.CarpSpec{
							NodeSelector: map[string]string{},
						},
					},
				},
			},
			expect: "azhzAAoECgASABJBCgYKABIAGgASNwoQCgASABoAIgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgAaACIA",
		},
		{
			name: "List just kind",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind: "List",
				},
			},
			expect: "azhzAAoICgASBExpc3QSCAoGCgASABoAGgAiAA==",
		},
		{
			name: "List just apiVersion",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
				},
			},
			expect: "azhzAAoGCgJ2MRIAEggKBgoAEgAaABoAIgA=",
		},
		{
			name: "List no elements",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "2345",
				},
				Items: []testapigroupv1.Carp{},
			},
			expect: "azhzAAoKCgJ2MRIETGlzdBIMCgoKABIEMjM0NRoAGgAiAA==",
		},
		{
			name: "List one element with continue",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion:    "2345",
					Continue:           "abc",
					RemainingItemCount: &remainingItems,
				},
				Items: []testapigroupv1.Carp{
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod",
						Namespace: "default",
					}},
				},
			},
			expect: "azhzAAoKCgJ2MRIETGlzdBJUCg8KABIEMjM0NRoDYWJjIAESQQoaCgNwb2QSABoHZGVmYXVsdCIAKgAyADgAQgASFxoAQgBKAFIAWABgAGgAggEAigEAmgEAGgoKABoAIgAqADIAGgAiAA==",
		},
		{
			name: "List two elements",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "2345",
				},
				Items: []testapigroupv1.Carp{
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod",
						Namespace: "default",
					}},
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "default2",
					}},
				},
			},
			expect: "azhzAAoKCgJ2MRIETGlzdBKUAQoKCgASBDIzNDUaABJBChoKA3BvZBIAGgdkZWZhdWx0IgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgASQwocCgRwb2QyEgAaCGRlZmF1bHQyIgAqADIAOABCABIXGgBCAEoAUgBYAGAAaACCAQCKAQCaAQAaCgoAGgAiACoAMgAaACIA",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := s.Encode(tc.in, &buf); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(base64.StdEncoding.EncodeToString(buf.Bytes()), tc.expect); diff != "" {
				t.Errorf("not matching:\n%s", diff)
			}
		})
	}
}
