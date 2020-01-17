/*
Copyright 2018 The Kubernetes Authors.

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

package merge

import (
	"fmt"
	"sort"
	"strings"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// Conflict is a conflict on a specific field with the current manager of
// that field. It does implement the error interface so that it can be
// used as an error.
type Conflict struct {
	Manager string
	Path    fieldpath.Path
}

// Conflict is an error.
var _ error = Conflict{}

// Error formats the conflict as an error.
func (c Conflict) Error() string {
	return fmt.Sprintf("conflict with %q: %v", c.Manager, c.Path)
}

// Equals returns true if c == c2
func (c Conflict) Equals(c2 Conflict) bool {
	if c.Manager != c2.Manager {
		return false
	}
	return c.Path.Equals(c2.Path)
}

// Conflicts accumulates multiple conflicts and aggregates them by managers.
type Conflicts []Conflict

var _ error = Conflicts{}

// Error prints the list of conflicts, grouped by sorted managers.
func (conflicts Conflicts) Error() string {
	if len(conflicts) == 1 {
		return conflicts[0].Error()
	}

	m := map[string][]fieldpath.Path{}
	for _, conflict := range conflicts {
		m[conflict.Manager] = append(m[conflict.Manager], conflict.Path)
	}

	managers := []string{}
	for manager := range m {
		managers = append(managers, manager)
	}

	// Print conflicts by sorted managers.
	sort.Strings(managers)

	messages := []string{}
	for _, manager := range managers {
		messages = append(messages, fmt.Sprintf("conflicts with %q:", manager))
		for _, path := range m[manager] {
			messages = append(messages, fmt.Sprintf("- %v", path))
		}
	}
	return strings.Join(messages, "\n")
}

// Equals returns true if the lists of conflicts are the same.
func (c Conflicts) Equals(c2 Conflicts) bool {
	if len(c) != len(c2) {
		return false
	}
	for i := range c {
		if !c[i].Equals(c2[i]) {
			return false
		}
	}
	return true
}

// ConflictsFromManagers creates a list of conflicts given Managers sets.
func ConflictsFromManagers(sets fieldpath.ManagedFields) Conflicts {
	conflicts := []Conflict{}

	for manager, set := range sets {
		set.Set().Iterate(func(p fieldpath.Path) {
			conflicts = append(conflicts, Conflict{
				Manager: manager,
				Path:    p,
			})
		})
	}

	return conflicts
}
