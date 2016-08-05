/*
Copyright 2016 The Kubernetes Authors.

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

package tolerations

import (
	"k8s.io/kubernetes/pkg/api"
)

// DoTolerationsSatisfyWhilelist checks if the provided tolerations
// satisfy the provided whitelist and returns true, otherwise returns false
func DoTolerationsSatisfyWhitelist(tolerations []api.Toleration, nsWhiteList []api.Toleration) bool {
	if len(nsWhiteList) == 0 {
		return true
	}

	for i := range tolerations {
		found := false
		for j := range nsWhiteList {
			if tolerations[i].Key == nsWhiteList[j].Key &&
				tolerations[i].Operator == nsWhiteList[j].Operator &&
				tolerations[i].Value == nsWhiteList[j].Value &&
				tolerations[i].Effect == nsWhiteList[j].Effect {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// ConflictingTolerations returns true if the key of two tolerations match
// but one or more other fields differ, otherwise returns false
func ConflictingTolerations(first []api.Toleration, second []api.Toleration) bool {
	for i := range first {
		for j := range second {
			if first[i].Key == second[j].Key {
				if first[i].Operator != second[j].Operator ||
					first[i].Value != second[j].Value ||
					first[i].Effect != second[j].Effect {
					return true
				}
			}
		}
	}
	return false
}

// MergeTolerations merges two sets of tolerations into one
// it does not check for conflicts
// it assumes no duplicates in individual set of tolerations
func MergeTolerations(first []api.Toleration, second []api.Toleration) []api.Toleration {
	var mergedTolerations []api.Toleration
	mergedTolerations = append(mergedTolerations, second...)
	for i := range first {
		found := false
		for j := range second {
			if first[i].Key == second[j].Key {
				found = true
				break
			}
		}
		if !found {
			mergedTolerations = append(mergedTolerations, first[i])
		}
	}
	return mergedTolerations
}

// EqualTolerations returns true if two sets of tolerations are equal, otherwise false
// it assumes no duplicates in individual set of tolerations
func EqualTolerations(first []api.Toleration, second []api.Toleration) bool {
	if len(first) != len(second) {
		return false
	}
	for i := range first {
		found := false
		for j := range second {
			if first[i].Key == second[j].Key &&
				first[i].Operator == second[j].Operator &&
				first[i].Value == second[j].Value &&
				first[i].Effect == second[j].Effect {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
