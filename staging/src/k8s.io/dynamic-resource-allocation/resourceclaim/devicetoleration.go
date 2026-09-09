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

package resourceclaim

import (
	resourceapi "k8s.io/api/resource/v1"
)

// ToleratesTaint checks if the toleration tolerates the taint.
// The matching follows the rules below:
//
//  1. Empty toleration.effect means to match all taint effects,
//     otherwise taint effect must equal to toleration.effect.
//  2. If toleration.operator is 'Exists', it means to match all taint values.
//  3. Empty toleration.key means to match all taint keys.
//     If toleration.key is empty, toleration.operator must be 'Exists';
//     this combination means to match all taint values and all taint keys.
//
// Callers must check separately what the effect is and only react to
// known effects. Unknown effects and the None effect must be ignored.
func ToleratesTaint(toleration resourceapi.DeviceToleration, taint resourceapi.DeviceTaint) bool {
	// This code was copied from https://github.com/kubernetes/kubernetes/blob/f007012f5fe49e40ae0596cf463a8e7b247b3357/staging/src/k8s.io/api/core/v1/toleration.go#L39-L56.
	// It wasn't placed in the resourceapi package because code related to logic
	// doesn't belong there.
	if toleration.Effect != "" && toleration.Effect != taint.Effect {
		return false
	}

	if toleration.Key != "" && toleration.Key != taint.Key {
		return false
	}

	switch toleration.Operator {
	// Empty operator means Equal, should be set because of defaulting.
	case "", resourceapi.DeviceTolerationOpEqual:
		return toleration.Value == taint.Value
	case resourceapi.DeviceTolerationOpExists:
		return true
	default:
		return false
	}
}
