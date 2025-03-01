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
	"io"
	"os/exec"
	"testing"

	"github.com/google/go-cmp/cmp"

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
			actualBytes := buf.Bytes()
			expectBytes, err := io.ReadAll(base64.NewDecoder(base64.StdEncoding, bytes.NewBufferString(tc.expect)))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(expectBytes, actualBytes) {
				t.Errorf("expected:\n%s\ngot:\n%s", tc.expect, base64.StdEncoding.EncodeToString(actualBytes))
				t.Log(cmp.Diff(dumpProto(t, actualBytes[4:]), dumpProto(t, expectBytes[4:])))
			}
		})
	}
}

// dumpProto does a best-effort dump of the given proto bytes using protoc if it can be found in the path.
// This is only used when the test has already failed, to try to give more visibility into the diff of the failure.
func dumpProto(t *testing.T, data []byte) string {
	t.Helper()
	protoc, err := exec.LookPath("protoc")
	if err != nil {
		t.Logf("cannot find protoc in path to dump proto contents: %v", err)
		return ""
	}
	cmd := exec.Command(protoc, "--decode_raw")
	cmd.Stdin = bytes.NewBuffer(data)
	d, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("protoc invocation failed: %v", err)
		return ""
	}
	return string(d)
}
