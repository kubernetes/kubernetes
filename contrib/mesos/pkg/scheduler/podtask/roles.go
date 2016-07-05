/*
Copyright 2015 The Kubernetes Authors.

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

package podtask

// rolePredicate is a predicate function on role strings
type rolePredicate func(string) bool

// filterRoles filters the given slice of roles and returns a slice of roles
// matching all given predicates
func filterRoles(roles []string, ps ...rolePredicate) []string {
	filtered := make([]string, 0, len(roles))

next:
	for _, r := range roles {
		for _, p := range ps {
			if !p(r) {
				continue next
			}
		}

		filtered = append(filtered, r)
	}

	return filtered
}

// seenRole returns a rolePredicate which returns true
// if a given role has already been seen in previous invocations.
func seenRole() rolePredicate {
	seen := map[string]struct{}{}

	return func(role string) bool {
		_, ok := seen[role]

		if !ok {
			seen[role] = struct{}{}
		}

		return ok
	}
}

// emptyRole returns true if the given role is empty
func emptyRole(name string) bool {
	return name == ""
}

// not returns a rolePredicate which returns the negation
// of the given predicate
func not(p rolePredicate) rolePredicate {
	return func(r string) bool {
		return !p(r)
	}
}

// inRoles returns a rolePredicate which returns true
// if the given role is present in the given roles
func inRoles(roles ...string) rolePredicate {
	roleSet := make(map[string]struct{}, len(roles))

	for _, r := range roles {
		roleSet[r] = struct{}{}
	}

	return func(r string) bool {
		_, ok := roleSet[r]
		return ok
	}
}
