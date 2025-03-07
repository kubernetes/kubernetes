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

package v1beta1

// MatchDeviceToleration checks if the two tolerations have the same <key,effect,operator,value> combination.
// TODO: uniqueness check for tolerations in api validations.
func (t DeviceToleration) Matches(o DeviceToleration) bool {
	return t.Key == o.Key &&
		t.Effect == o.Effect &&
		t.Operator == o.Operator &&
		t.Value == o.Value
}

// ToleratesTaint checks if the toleration tolerates the taint.
// The matching follows the rules below:
//
//  1. Empty toleration.effect means to match all taint effects,
//     otherwise taint effect must equal to toleration.effect.
//  2. If toleration.operator is 'Exists', it means to match all taint values.
//  3. Empty toleration.key means to match all taint keys.
//     If toleration.key is empty, toleration.operator must be 'Exists';
//     this combination means to match all taint values and all taint keys.
func (t DeviceToleration) ToleratesTaint(taint DeviceTaintAtom) bool {
	if t.Effect != "" && t.Effect != taint.Effect {
		return false
	}

	if t.Key != "" && t.Key != taint.Key {
		return false
	}

	// TODO: Use proper defaulting?
	switch t.Operator {
	// empty operator means Equal
	case "", DeviceTolerationOpEqual:
		return t.Value == taint.Value
	case DeviceTolerationOpExists:
		return true
	default:
		return false
	}
}
