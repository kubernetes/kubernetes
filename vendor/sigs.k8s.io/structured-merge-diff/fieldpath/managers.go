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

package fieldpath

import (
	"fmt"
	"strings"
)

// APIVersion describes the version of an object or of a fieldset.
type APIVersion string

type VersionedSet interface {
	Set() *Set
	APIVersion() APIVersion
	Applied() bool
}

// VersionedSet associates a version to a set.
type versionedSet struct {
	set        *Set
	apiVersion APIVersion
	applied    bool
}

func NewVersionedSet(set *Set, apiVersion APIVersion, applied bool) VersionedSet {
	return versionedSet{
		set:        set,
		apiVersion: apiVersion,
		applied:    applied,
	}
}

func (v versionedSet) Set() *Set {
	return v.set
}

func (v versionedSet) APIVersion() APIVersion {
	return v.apiVersion
}

func (v versionedSet) Applied() bool {
	return v.applied
}

// ManagedFields is a map from manager to VersionedSet (what they own in
// what version).
type ManagedFields map[string]VersionedSet

// Difference returns a symmetric difference between two Managers. If a
// given user's entry has version X in lhs and version Y in rhs, then
// the return value for that user will be from rhs. If the difference for
// a user is an empty set, that user will not be inserted in the map.
func (lhs ManagedFields) Difference(rhs ManagedFields) ManagedFields {
	diff := ManagedFields{}

	for manager, left := range lhs {
		right, ok := rhs[manager]
		if !ok {
			if !left.Set().Empty() {
				diff[manager] = left
			}
			continue
		}

		// If we have sets in both but their version
		// differs, we don't even diff and keep the
		// entire thing.
		if left.APIVersion() != right.APIVersion() {
			diff[manager] = right
			continue
		}

		newSet := left.Set().Difference(right.Set()).Union(right.Set().Difference(left.Set()))
		if !newSet.Empty() {
			diff[manager] = NewVersionedSet(newSet, right.APIVersion(), false)
		}
	}

	for manager, set := range rhs {
		if _, ok := lhs[manager]; ok {
			// Already done
			continue
		}
		if !set.Set().Empty() {
			diff[manager] = set
		}
	}

	return diff
}

func (lhs ManagedFields) String() string {
	s := strings.Builder{}
	for k, v := range lhs {
		fmt.Fprintf(&s, "%s:\n", k)
		fmt.Fprintf(&s, "- Applied: %v\n", v.Applied())
		fmt.Fprintf(&s, "- APIVersion: %v\n", v.APIVersion())
		fmt.Fprintf(&s, "- Set: %v\n", v.Set())
	}
	return s.String()
}
