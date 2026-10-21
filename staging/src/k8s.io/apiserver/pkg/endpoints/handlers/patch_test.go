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

package handlers

import (
	"bytes"
	"encoding/json"
	"testing"

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/types"
)

func TestValidateAndTranscodePatch(t *testing.T) {
	tests := []struct {
		name      string
		patchType types.PatchType
		patch     []byte
		expected  []byte
		expectErr bool
	}{
		{
			name:      "JSONPatchType valid",
			patchType: types.JSONPatchType,
			patch:     []byte(`[{"op": "add", "path": "/a", "value": 1}]`),
			expected:  []byte(`[{"op": "add", "path": "/a", "value": 1}]`),
			expectErr: false,
		},
		{
			name:      "JSONPatchType invalid",
			patchType: types.JSONPatchType,
			patch:     []byte(`[{"op": "add", "path": "/a", "value": 1`),
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "MergePatchType valid",
			patchType: types.MergePatchType,
			patch:     []byte(`{"a": 1}`),
			expected:  []byte(`{"a": 1}`),
			expectErr: false,
		},
		{
			name:      "MergePatchType invalid",
			patchType: types.MergePatchType,
			patch:     []byte(`{"a": 1`),
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "StrategicMergePatchType valid",
			patchType: types.StrategicMergePatchType,
			patch:     []byte(`{"a": 1}`),
			expected:  []byte(`{"a": 1}`),
			expectErr: false,
		},
		{
			name:      "StrategicMergePatchType invalid",
			patchType: types.StrategicMergePatchType,
			patch:     []byte(`{"a": 1`),
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "ApplyCBORPatchType valid",
			patchType: types.ApplyCBORPatchType,
			patch: func() []byte {
				data := map[string]any{"a": 1}
				b, _ := cbor.Marshal(data)
				return b
			}(),
			expected:  []byte(`{"a":1}`),
			expectErr: false,
		},
		{
			name:      "ApplyCBORPatchType invalid",
			patchType: types.ApplyCBORPatchType,
			patch:     []byte{0xff, 0xff},
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "ApplyYAMLPatchType valid",
			patchType: types.ApplyYAMLPatchType,
			patch:     []byte("a: 1"),
			expected:  []byte(`{"a":1}`),
			expectErr: false,
		},
		{
			name:      "ApplyYAMLPatchType invalid",
			patchType: types.ApplyYAMLPatchType,
			patch:     []byte(":\n  invalid"),
			expected:  nil,
			expectErr: true,
		},
		{
			name:      "Empty patch bytes",
			patchType: types.ApplyCBORPatchType,
			patch:     []byte{},
			expected:  nil,
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := validateAndTranscodePatch(tc.patch, tc.patchType)
			if (err != nil) != tc.expectErr {
				t.Fatalf("expected error: %v, got: %v", tc.expectErr, err)
			}
			if tc.expectErr {
				return
			}

			// Normalize JSON to compare
			if tc.patchType == types.ApplyCBORPatchType || tc.patchType == types.ApplyYAMLPatchType {
				var gotMap, expectedMap any
				if err := json.Unmarshal(got, &gotMap); err != nil {
					t.Fatalf("failed to unmarshal got as JSON: %v. raw got: %s", err, string(got))
				}
				if err := json.Unmarshal(tc.expected, &expectedMap); err != nil {
					t.Fatalf("failed to unmarshal expected as JSON: %v", err)
				}
				gotNorm, _ := json.Marshal(gotMap)
				expectedNorm, _ := json.Marshal(expectedMap)
				if !bytes.Equal(gotNorm, expectedNorm) {
					t.Errorf("expected transcoded JSON:\n%s\ngot:\n%s", string(expectedNorm), string(gotNorm))
				}
			} else if !bytes.Equal(got, tc.expected) {
				t.Errorf("expected unchanged bytes:\n%s\ngot:\n%s", string(tc.expected), string(got))
			}
		})
	}
}
