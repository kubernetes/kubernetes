/*
Copyright 2017 The Kubernetes Authors.

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
	api "k8s.io/kubernetes/pkg/apis/core"
)

type key struct {
	tolerationKey string
	effect        api.TaintEffect
}

func convertTolerationToKey(in api.Toleration) key {
	return key{in.Key, in.Effect}
}

// VerifyAgainstWhitelist checks if the provided tolerations
// satisfy the provided whitelist and returns true, otherwise returns false
func VerifyAgainstWhitelist(tolerations []api.Toleration, whitelist []api.Toleration) bool {
	if len(whitelist) == 0 {
		return true
	}

	t := ConvertTolerationToAMap(tolerations)
	w := ConvertTolerationToAMap(whitelist)

	for k1, v1 := range t {
		if v2, ok := w[k1]; !ok || !AreEqual(v1, v2) {
			return false
		}
	}
	return true
}

// IsConflict returns true if the key of two tolerations match
// but one or more other fields differ, otherwise returns false
func IsConflict(first []api.Toleration, second []api.Toleration) bool {
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
func MergeTolerations(first []api.Toleration, second []api.Toleration) []api.Toleration {
	var mergedTolerations []api.Toleration
	mergedTolerations = append(mergedTolerations, second...)
	firstMap := ConvertTolerationToAMap(first)
	secondMap := ConvertTolerationToAMap(second)
	// preserve order of first when merging
	for _, v := range first {
		k := convertTolerationToKey(v)
		// if first contains key conflicts, the last one takes precedence
		if _, ok := secondMap[k]; !ok && firstMap[k] == v {
			mergedTolerations = append(mergedTolerations, v)
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
func ConvertTolerationToAMap(in []api.Toleration) map[key]api.Toleration {
	out := map[key]api.Toleration{}
	for _, v := range in {
		out[convertTolerationToKey(v)] = v
	}
	return out
}

// AreEqual checks if two provided tolerations are equal or not.
func AreEqual(first, second api.Toleration) bool {
	if first.Key == second.Key &&
		first.Operator == second.Operator &&
		first.Value == second.Value &&
		first.Effect == second.Effect &&
		AreTolerationSecondsEqual(first.TolerationSeconds, second.TolerationSeconds) {
		return true
	}
	return false
}

// AreTolerationSecondsEqual checks if two provided TolerationSeconds are equal or not.
func AreTolerationSecondsEqual(ts1, ts2 *int64) bool {
	if ts1 == ts2 {
		return true
	}
	if ts1 != nil && ts2 != nil && *ts1 == *ts2 {
		return true
	}
	return false
}
