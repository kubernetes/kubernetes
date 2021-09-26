/*
Copyright 2019 The Kubernetes Authors.

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

package merge_test

import (
	"fmt"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// specificVersionConverter doesn't convert and return the exact same
// object, but only for versions that are explicitely listed.
type specificVersionConverter struct {
	AcceptedVersions []fieldpath.APIVersion
}

func (d *specificVersionConverter) Convert(object *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	for _, v := range d.AcceptedVersions {
		if v == version {
			return object, nil
		}
	}
	return nil, fmt.Errorf("Unknown version: %v", version)
}

func (d *specificVersionConverter) IsMissingVersionError(err error) bool {
	return err != nil
}

// Managers of fields in a version that no longer exist are
// automatically removed. Make sure this works as intended.
func TestObsoleteVersions(t *testing.T) {
	converter := &specificVersionConverter{
		AcceptedVersions: []fieldpath.APIVersion{"v1", "v2"},
	}
	state := State{
		Updater: &merge.Updater{Converter: converter},
		Parser:  DeducedParser,
	}

	if err := state.Update(typed.YAMLObject(`{"v1": 0}`), fieldpath.APIVersion("v1"), "v1"); err != nil {
		t.Fatalf("Failed to apply: %v", err)
	}
	if err := state.Update(typed.YAMLObject(`{"v1": 0, "v2": 0}`), fieldpath.APIVersion("v2"), "v2"); err != nil {
		t.Fatalf("Failed to apply: %v", err)
	}
	// Remove v1, add v3 instead.
	converter.AcceptedVersions = []fieldpath.APIVersion{"v2", "v3"}

	if err := state.Update(typed.YAMLObject(`{"v1": 0, "v2": 0, "v3": 0}`), fieldpath.APIVersion("v3"), "v3"); err != nil {
		t.Fatalf("Failed to apply: %v", err)
	}

	managers := fieldpath.ManagedFields{
		"v2": fieldpath.NewVersionedSet(
			_NS(
				_P("v2"),
			),
			"v2",
			false,
		),
		"v3": fieldpath.NewVersionedSet(
			_NS(
				_P("v3"),
			),
			"v3",
			false,
		),
	}
	if diff := state.Managers.Difference(managers); len(diff) != 0 {
		t.Fatalf("expected Managers to be %v, got %v", managers, state.Managers)
	}
}

func TestApplyObsoleteVersion(t *testing.T) {
	converter := &specificVersionConverter{
		AcceptedVersions: []fieldpath.APIVersion{"v1"},
	}
	tparser, err := typed.NewParser(`types:
- name: sets
  map:
    fields:
    - name: list
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: associative`)
	if err != nil {
		t.Fatalf("Failed to create parser: %v", err)
	}
	parser := SameVersionParser{T: tparser.Type("sets")}
	state := State{
		Updater: &merge.Updater{Converter: converter},
		Parser:  SameVersionParser{T: parser.Type("sets")},
	}

	if err := state.Apply(typed.YAMLObject(`{"list": ["a", "b", "c", "d"]}`), fieldpath.APIVersion("v1"), "apply", false); err != nil {
		t.Fatalf("Failed to apply: %v", err)
	}
	// Remove v1, add v2 instead.
	converter.AcceptedVersions = []fieldpath.APIVersion{"v2"}

	if err := state.Apply(typed.YAMLObject(`{"list": ["a"]}`), fieldpath.APIVersion("v2"), "apply", false); err != nil {
		t.Fatalf("Failed to apply: %v", err)
	}

	comparison, err := state.CompareLive(`{"list": ["a", "b", "c", "d"]}`, "v2")
	if err != nil {
		t.Fatalf("Failed to compare live object: %v", err)
	}
	if !comparison.IsSame() {
		t.Fatalf("Unexpected object:\n%v", comparison)
	}
}
