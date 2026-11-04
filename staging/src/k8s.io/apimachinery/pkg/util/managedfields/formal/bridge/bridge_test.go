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

package bridge

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestIndependentGoCorpus(t *testing.T) {
	data, err := os.ReadFile("../testdata/bridge.yaml")
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := ReadCorpus(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckGo(corpus.Cases); err != nil {
		t.Fatal(err)
	}
}

func TestComparatorRejectsRegressionsAndStaleExceptions(t *testing.T) {
	expected := Expected{Object: map[string]any{"x": "one"}, Owners: map[string][]string{"A": {".x"}}}
	snapshot, err := expectedSnapshot(expected)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name     string
		step     Step
		goResult Snapshot
	}{
		{"changed value", Step{}, Snapshot{"", map[string]string{".x": "value:\"two\""}, snapshot.Owners}},
		{"lost owner", Step{}, Snapshot{"", snapshot.Cells, map[string][]string{}}},
		{"different error", Step{}, Snapshot{"conflict", snapshot.Cells, snapshot.Owners}},
		{"stale exception", Step{Expect: &expected, GoExpect: &expected, Difference: "old discrepancy"}, snapshot},
		{"unreviewed exception", Step{Difference: "any difference"}, Snapshot{"conflict", snapshot.Cells, snapshot.Owners}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := compareStep(tt.step, snapshot, tt.goResult); err == nil {
				t.Fatal("comparison accepted a regression or unchecked difference")
			}
		})
	}
}

func TestCorpusRejectsMissingOraclesAndBroadExceptions(t *testing.T) {
	for _, data := range []string{
		"version: 1\ncases: []\n",
		"version: 1\ncases: [{name: missing, steps: [{manager: A, apply: {}}]}]\n",
		"version: 1\nunknown: true\ncases: []\n",
		"version: 1\ncases: [{name: unchecked, steps: [{manager: A, apply: {}, expect: {object: {}, owners: {}}, difference: anything}]}]\n",
	} {
		if _, err := ReadCorpus([]byte(data)); err == nil {
			t.Fatalf("accepted invalid corpus: %s", data)
		}
	}
}

func TestGeneratedFailurePrefixCanBeReplayed(t *testing.T) {
	cases := GeneratedScalarCases(336, 1, 3)
	data, err := json.Marshal(Corpus{Version: 1, Cases: cases})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReadCorpus(data); err == nil {
		t.Fatal("ordinary corpus accepted missing independent expectations")
	}
	reproducer, err := ReadReproducer(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(reproducer.Cases) != 1 || len(reproducer.Cases[0].Steps) != 3 {
		t.Fatal("reproducer lost requests")
	}
}

func TestAdapterPreservesEmptyValuesAndRejectsDuplicateKeys(t *testing.T) {
	for _, object := range []map[string]any{
		{"items": []any{}},
		{"settings": map[string]any{}},
		{"atomic": map[string]any{}},
	} {
		cells, _, err := flatten(object, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(cells) != 1 {
			t.Fatalf("lost explicit empty value: %#v", object)
		}
	}
	_, _, err := flatten(map[string]any{"items": []any{
		map[string]any{"name": "same"}, map[string]any{"name": "same"},
	}}, nil)
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate associative key was not rejected: %v", err)
	}
}

func TestNormalizerEncodesAtomicMapsAsLeaves(t *testing.T) {
	c := Case{Name: "atomic", Steps: []Step{{Manager: "A", Apply: map[string]any{
		"atomic": map[string]any{"left": "one", "right": "two"},
	}}}}
	fixture, err := normalize(c)
	if err != nil {
		t.Fatal(err)
	}
	r := fixture.Steps[0].Request
	if len(r.Writes) != 1 || r.Writes[0].Content.Kind != "value" || r.Desired[0].Kind != "value" {
		t.Fatalf("atomic map was split into child claims: %#v", r)
	}
	encoded, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(encoded) {
		t.Fatal("invalid Lean fixture JSON")
	}
}

func TestStructuralOwnershipAcrossApplyAndUpdate(t *testing.T) {
	initial := map[string]any{
		"number": 1.5, "flag": true, "nullable": nil,
		"entries":  map[string]any{"one": map[string]any{"left": "x"}},
		"services": []any{map[string]any{"name": "n", "port": float64(80), "config": map[string]any{"left": "x"}}},
		"tags":     []any{"green"},
	}
	replacement := map[string]any{
		"number": 2.5, "flag": true, "nullable": nil,
		"services": []any{map[string]any{"name": "n", "port": float64(80), "config": map[string]any{"left": "x"}}},
		"tags":     []any{"green"},
	}
	owned := []string{".flag", ".nullable", `.services[name="n",port=80]`, `.services[name="n",port=80].config.left`, `.tags[="green"]`}
	first := append(append([]string{}, owned...), ".number", ".entries.one", ".entries.one.left")
	c := Case{Name: "structural ownership", Schema: structuralYAML, Semantics: "current", Steps: []Step{
		{Manager: "A", Apply: initial, Expect: &Expected{Object: initial, Owners: map[string][]string{"A": first}}},
		{Manager: "U", Operation: "update", Apply: replacement, Expect: &Expected{
			Object: replacement, Owners: map[string][]string{"A": owned, "U@update": {".number"}},
		}},
		{Manager: "A", Apply: map[string]any{}, Expect: &Expected{
			Object: map[string]any{"number": 2.5}, Owners: map[string][]string{"U@update": {".number"}},
		}},
	}}
	if err := CheckGo([]Case{c}); err != nil {
		t.Fatal(err)
	}
	if _, err := normalize(c); err != nil {
		t.Fatal(err)
	}
}
