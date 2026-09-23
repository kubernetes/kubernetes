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

package internal

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

func TestOwnsManagedFields(t *testing.T) {
	set := func(paths ...fieldpath.Path) *fieldpath.Set { return fieldpath.NewSet(paths...) }
	tests := []struct {
		name string
		sets []*fieldpath.Set
		want bool
	}{
		{name: "none"},
		{name: "other metadata", sets: []*fieldpath.Set{set(fieldpath.MakePathOrDie("metadata", "labels", "a"), fieldpath.MakePathOrDie("spec"))}},
		{name: "managedFields outside metadata", sets: []*fieldpath.Set{set(fieldpath.MakePathOrDie("spec", "managedFields"))}},
		{name: "managedFields", sets: []*fieldpath.Set{set(fieldpath.MakePathOrDie("spec")), set(fieldpath.MakePathOrDie("metadata", "managedFields"))}, want: true},
		{name: "within managedFields", sets: []*fieldpath.Set{set(fieldpath.MakePathOrDie("metadata", "managedFields", 0, "manager"))}, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			managers := fieldpath.ManagedFields{}
			for i, s := range tc.sets {
				managers[string(rune('a'+i))] = fieldpath.NewVersionedSet(s, "v1", false)
			}
			if got := ownsManagedFields(managers); got != tc.want {
				t.Errorf("ownsManagedFields() = %v, want %v", got, tc.want)
			}
		})
	}
}
