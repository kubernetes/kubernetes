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

package watchgroup

// Reassignment represents a change in resource ownership between two members.
type Reassignment struct {
	Key      string
	OldOwner MemberID
	NewOwner MemberID
}

// ComputeReassignments compares old and new hash rings and identifies which
// resource keys changed ownership. Only keys whose owner differs between the
// two rings are returned.
func ComputeReassignments(oldRing, newRing *HashRing, allKeys []string) []Reassignment {
	var result []Reassignment
	for _, key := range allKeys {
		oldOwner := oldRing.GetNode(key)
		newOwner := newRing.GetNode(key)
		if oldOwner != newOwner {
			result = append(result, Reassignment{
				Key:      key,
				OldOwner: oldOwner,
				NewOwner: newOwner,
			})
		}
	}
	return result
}
