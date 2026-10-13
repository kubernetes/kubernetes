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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// VerifyAgainstWhitelist checks if the provided tolerations
// satisfy the provided whitelist and returns true, otherwise returns false
func VerifyAgainstWhitelist(tolerations, whitelist []api.Toleration) bool {
	if len(whitelist) == 0 || len(tolerations) == 0 {
		return true
	}

next:
	for _, t := range tolerations {
		for _, w := range whitelist {
			if isSuperset(w, t) {
				continue next
			}
		}
		return false
	}

	return true
}

// MergeTolerations merges two sets of tolerations into one. If one toleration is a superset of
// another, only the superset is kept.
func MergeTolerations(first, second []api.Toleration) []api.Toleration {
	all := append(first, second...)
	var merged []api.Toleration

next:
	for i, t := range all {
		for _, t2 := range merged {
			if isSuperset(t2, t) {
				continue next // t is redundant; ignore it
			}
		}
		if i+1 < len(all) {
			for _, t2 := range all[i+1:] {
				// If the tolerations are equal, prefer the first.
				if !apiequality.Semantic.DeepEqual(&t, &t2) && isSuperset(t2, t) {
					continue next // t is redundant; ignore it
				}
			}
		}
		merged = append(merged, t)
	}

	return merged
}

// isSuperset checks whether ss tolerates a superset of t.
func isSuperset(ss, t api.Toleration) bool {
	if apiequality.Semantic.DeepEqual(&t, &ss) {
		return true
	}

	if t.Key != ss.Key &&
		// An empty key with Exists operator means match all keys & values.
		(ss.Key != "" || ss.Operator != api.TolerationOpExists) {
		return false
	}

	// An empty effect means match all effects.
	if t.Effect != ss.Effect && ss.Effect != "" {
		return false
	}

	if ss.Effect == api.TaintEffectNoExecute {
		if ss.TolerationSeconds != nil {
			if t.TolerationSeconds == nil ||
				*t.TolerationSeconds > *ss.TolerationSeconds {
				return false
			}
		}
	}

	switch ss.Operator {
	case api.TolerationOpEqual, "": // empty operator means Equal
		return t.Operator == api.TolerationOpEqual && t.Value == ss.Value
	case api.TolerationOpExists:
		return true
	default:
		klog.Errorf("Unknown toleration operator: %s", ss.Operator)
		return false
	}
}
