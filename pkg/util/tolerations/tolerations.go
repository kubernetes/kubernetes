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

	firstMap := ConvertTolerationToAMap(tolerations)
	secondMap := ConvertTolerationToAMap(nsWhiteList)

	for k1, v1 := range firstMap {
		if v2, ok := secondMap[k1]; !ok || !AreEqual(v1, v2) {
			return false
		}
	}
	return true
}

// ConflictingTolerations returns true if the key of two tolerations match
// but one or more other fields differ, otherwise returns false
func ConflictingTolerations(first []api.Toleration, second []api.Toleration) bool {
	firstMap := ConvertTolerationToAMap(first)
	secondMap := ConvertTolerationToAMap(second)

	for k1, v1 := range firstMap {
		if v2, ok := secondMap[k1]; ok && !AreEqual(v1, v2) {
			return true
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

	firstMap := ConvertTolerationToAMap(first)
	secondMap := ConvertTolerationToAMap(second)

	for k1, v1 := range firstMap {
		if _, ok := secondMap[k1]; !ok {
			mergedTolerations = append(mergedTolerations, v1)
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

	firstMap := ConvertTolerationToAMap(first)
	secondMap := ConvertTolerationToAMap(second)

	for k1, v1 := range firstMap {
		if v2, ok := secondMap[k1]; !ok || !AreEqual(v1, v2) {
			return false
		}
	}
	return true
}

// ConvertTolerationToAMap converts toleration list into a map[string]api.Toleration
func ConvertTolerationToAMap(in []api.Toleration) map[string]api.Toleration {
	out := map[string]api.Toleration{}
	for i := range in {
		out[in[i].Key] = in[i]
	}
	return out
}

// AreEqual checks if two provided tolerations are equal or not.
func AreEqual(first, second api.Toleration) bool {
	if first.Key == second.Key &&
		first.Operator == second.Operator &&
		first.Value == second.Value &&
		first.Effect == second.Effect {
		return true
	}
	return false
}
