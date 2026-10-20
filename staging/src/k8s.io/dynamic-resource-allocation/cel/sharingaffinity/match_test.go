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

package sharingaffinity

import (
	"encoding/json"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
)

func TestMetadataOf_Fields(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		want   configMetadata
	}{
		{"nil", nil, configMetadata{}},
		{"empty", map[string]any{}, configMetadata{}},
		{"v1", map[string]any{"apiVersion": "example.com/v1", "kind": "Config"}, configMetadata{group: "example.com", kind: "Config", hasGroup: true, hasKind: true}},
		{"v2", map[string]any{"apiVersion": "example.com/v2", "kind": "Config"}, configMetadata{group: "example.com", kind: "Config", hasGroup: true, hasKind: true}},
		{"groupless", map[string]any{"apiVersion": "v1"}, configMetadata{}},
		{"empty version", map[string]any{"apiVersion": ""}, configMetadata{}},
		{"empty group", map[string]any{"apiVersion": "/v1"}, configMetadata{}},
		{"last slash", map[string]any{"apiVersion": "example.com/extra/v1"}, configMetadata{group: "example.com/extra", hasGroup: true}},
		{"ignore version suffix", map[string]any{"apiVersion": "example.com/"}, configMetadata{group: "example.com", hasGroup: true}},
		{"kind only", map[string]any{"kind": "Config"}, configMetadata{kind: "Config", hasKind: true}},
		{"integer version", map[string]any{"apiVersion": int64(5), "kind": "Config"}, configMetadata{kind: "Config", hasKind: true}},
		{"boolean kind", map[string]any{"apiVersion": "example.com/v1", "kind": true}, configMetadata{group: "example.com", hasGroup: true}},
		{"nulls", map[string]any{"apiVersion": nil, "kind": nil}, configMetadata{}},
		{"structured metadata", map[string]any{"apiVersion": map[string]any{"group": "example.com"}, "kind": []any{"Config"}}, configMetadata{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := metadataOf(tc.params); got != tc.want {
				t.Errorf("metadataOf(%#v) = %#v, want %#v", tc.params, got, tc.want)
			}
		})
	}
}

func TestMatcher_Matrix(t *testing.T) {
	for _, tc := range []struct {
		name, group, kind, raw string
		want                   bool
	}{
		{"exact v1", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Config"}`, true},
		{"exact v2", "example.com", "Config", `{"apiVersion":"example.com/v2","kind":"Config"}`, true},
		{"wrong group", "example.com", "Config", `{"apiVersion":"other.example.com/v1","kind":"Config"}`, false},
		{"group case mismatch", "example.com", "Config", `{"apiVersion":"EXAMPLE.com/v1","kind":"Config"}`, false},
		{"wrong kind", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Other"}`, false},
		{"kind case mismatch", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"config"}`, false},
		{"group wildcard", "*", "Config", `{"apiVersion":"other.example.com/v1","kind":"Config"}`, true},
		{"group wildcard still checks kind", "*", "Config", `{"kind":"Other"}`, false},
		{"kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1","kind":"Other"}`, true},
		{"kind wildcard still checks group", "example.com", "*", `{"apiVersion":"other.example.com/v1"}`, false},
		{"both wildcards", "*", "*", `{"apiVersion":"other.example.com/v1","kind":"Other"}`, true},
		{"no metadata wildcards", "*", "*", `{}`, true},
		{"missing apiVersion exact", "example.com", "Config", `{"kind":"Config"}`, false},
		{"missing apiVersion wildcard", "*", "Config", `{"kind":"Config"}`, true},
		{"groupless exact", "example.com", "Config", `{"apiVersion":"v1","kind":"Config"}`, false},
		{"groupless wildcard", "*", "Config", `{"apiVersion":"v1","kind":"Config"}`, true},
		{"missing kind exact", "example.com", "Config", `{"apiVersion":"example.com/v1"}`, false},
		{"missing kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1"}`, true},
		{"numeric apiVersion exact", "example.com", "Config", `{"apiVersion":5,"kind":"Config"}`, false},
		{"numeric apiVersion wildcard", "*", "Config", `{"apiVersion":5,"kind":"Config"}`, true},
		{"boolean kind exact", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":true}`, false},
		{"boolean kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1","kind":true}`, true},
		{"null metadata exact", "example.com", "Config", `{"apiVersion":null,"kind":null}`, false},
		{"null metadata wildcard", "*", "*", `{"apiVersion":null,"kind":null}`, true},
		{"structured metadata wildcard", "*", "*", `{"apiVersion":{},"kind":[]}`, true},
		{"structured metadata exact", "example.com", "Config", `{"apiVersion":{},"kind":[]}`, false},
		{"empty metadata exact", "example.com", "Config", `{"apiVersion":"","kind":""}`, false},
		{"empty metadata wildcard", "*", "*", `{"apiVersion":"","kind":""}`, true},
		{"extra slash not valid group", "example.com", "Config", `{"apiVersion":"example.com/extra/v1","kind":"Config"}`, false},
		{"extra slash wildcard", "*", "Config", `{"apiVersion":"example.com/extra/v1","kind":"Config"}`, true},
		{"empty group exact", "example.com", "Config", `{"apiVersion":"/v1","kind":"Config"}`, false},
		{"empty group wildcard", "*", "Config", `{"apiVersion":"/v1","kind":"Config"}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var params map[string]any
			if err := json.Unmarshal([]byte(tc.raw), &params); err != nil {
				t.Fatalf("invalid fixture: %v", err)
			}
			if got := (matcher{apiGroup: tc.group, kind: tc.kind}).matches(metadataOf(params)); got != tc.want {
				t.Errorf("matches(%s) = %v, want %v", tc.raw, got, tc.want)
			}
			entry := testExtractor("subnet", `"blue"`)
			entry.APIGroup, entry.Kind = tc.group, tc.kind
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.want {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_NonmatchDoesNotEvaluate(t *testing.T) {
	entry := testExtractor("subnet", `parameters.mustExist`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	for _, raw := range []string{
		`{"apiVersion":"other.example.com/v1","kind":"Config"}`,
		`{"apiVersion":"example.com/v1","kind":"Other"}`,
		`{"apiVersion":"example.com/v1","kind":"config"}`,
		`{"kind":"Config"}`,
		`{"apiVersion":"v1","kind":"Config"}`,
		`{"apiVersion":"example.com/v1"}`,
		`{"apiVersion":7,"kind":true}`,
		`{}`,
	} {
		t.Run(raw, func(t *testing.T) {
			cfg := opaqueConfig(nil, raw)
			extractor := newTestExtractor()
			got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{cfg})
			if err != nil {
				t.Fatalf("metadata nonmatch evaluated missing-field expression: %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Unsatisfied: []string{"subnet"}})
			got, err = extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
				cfg,
				opaqueConfig(nil, `{"apiVersion":"example.com/v2","kind":"Config","mustExist":"blue"}`),
				cfg,
			})
			if err != nil {
				t.Fatalf("unrelated config prevented matching extraction: %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
		})
	}
}
