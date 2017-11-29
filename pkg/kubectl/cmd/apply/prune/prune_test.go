/*
Copyright 2014 The Kubernetes Authors.

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

package prune

import (
	"testing"

	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestParseGvks(t *testing.T) {

	// Create the fake RESTMapper from the fake factory.
	f, _, _, _ := cmdtesting.NewAPIFactory()
	mapper, _ := f.Object()

	tests := []struct {
		gvks     []string
		group    []string
		version  []string
		kind     []string
		resource []string
	}{
		{
			gvks:     []string{"core/v1/ConfigMap", "extensions/v1beta1/Deployment"},
			group:    []string{"", "extensions"},
			version:  []string{"v1", "v1beta1"},
			kind:     []string{"ConfigMap", "Deployment"},
			resource: []string{"configmaps", "deployments"},
		},
	}
	for _, test := range tests {
		mappings, err := ParseGvks(mapper, test.gvks)
		if err != nil {
			t.Errorf("Error--TestParseGvks: %v", err)
		}
		if len(test.gvks) != len(mappings) {
			t.Error("Error--TestParseGvks")
		}
		for i, mapping := range mappings {
			if mapping.GroupVersionKind.Group != test.group[i] {
				t.Errorf("Error--ParseGvks: %v", mapping.GroupVersionKind.Group)
			}
			if mapping.GroupVersionKind.Version != test.version[i] {
				t.Errorf("Error--ParseGvks: %v", mapping.GroupVersionKind.Version)
			}
			if mapping.GroupVersionKind.Kind != test.kind[i] {
				t.Errorf("Error--ParseGvks: %v", mapping.GroupVersionKind.Kind)
			}
			if mapping.Resource != test.resource[i] {
				t.Errorf("Error--ParseGvks: %v", mapping.Resource)
			}
		}
	}

	// Unknown group/version/kind should throw an error.
	_, err := ParseGvks(mapper, []string{"foo/bar/baz"})
	if err == nil {
		t.Error("Error: parsePruneWhitelst should return error when passed nil")
	}
}

func TestParsePruneWhitelist(t *testing.T) {

	tests := []struct {
		gvk    []string
		parsed [][]string
	}{
		{
			gvk:    []string{},
			parsed: [][]string{{}},
		},
		{
			gvk:    nil,
			parsed: [][]string{{}},
		},
		{
			gvk:    []string{"core/v1/ConfigMap"},
			parsed: [][]string{{"", "v1", "ConfigMap"}},
		},
		{
			gvk:    []string{"batch/v1beta1/Job"},
			parsed: [][]string{{"batch", "v1beta1", "Job"}},
		},
		{
			gvk: []string{
				"batch/v1beta1/Job",
				"core/v1/Pod",
				"extensions/v1beta1/Deployment",
			},
			parsed: [][]string{
				{"batch", "v1beta1", "Job"},
				{"", "v1", "Pod"},
				{"extensions", "v1beta1", "Deployment"},
			},
		},
	}

	for _, test := range tests {
		parsedPruneWhitelist, err := parsePruneWhitelist(test.gvk)
		if err != nil {
			t.Errorf("Error: parsePruneWhitelist: %v", err)
		}
		for i, parsedGvk := range parsedPruneWhitelist {
			for j, parsedField := range parsedGvk {
				if test.parsed[i][j] != parsedField {
					t.Errorf("Error: parsePruneWhitelist [expected/parsed]: [%v/%v]",
						test.parsed[i][j], parsedField)
				}
			}
		}
	}

}
