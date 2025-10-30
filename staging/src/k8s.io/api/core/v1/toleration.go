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

package v1

import (
	"strconv"
)

// MatchToleration checks if the toleration matches tolerationToMatch. Tolerations are unique by <key,effect,operator,value>,
// if the two tolerations have same <key,effect,operator,value> combination, regard as they match.
// TODO: uniqueness check for tolerations in api validations.
func (t *Toleration) MatchToleration(tolerationToMatch *Toleration) bool {
	return t.Key == tolerationToMatch.Key &&
		t.Effect == tolerationToMatch.Effect &&
		t.Operator == tolerationToMatch.Operator &&
		t.Value == tolerationToMatch.Value
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
//  4. If toleration.operator is 'Lt' or 'Gt', numeric comparison is performed
//     between toleration.value and taint.value.
func (t *Toleration) ToleratesTaint(taint *Taint) bool {
	if len(t.Effect) > 0 && t.Effect != taint.Effect {
		return false
	}

	if len(t.Key) > 0 && t.Key != taint.Key {
		return false
	}

	// TODO: Use proper defaulting when Toleration becomes a field of PodSpec
	switch t.Operator {
	// empty operator means Equal
	case "", TolerationOpEqual:
		return t.Value == taint.Value
	case TolerationOpExists:
		return true
	case TolerationOpLt, TolerationOpGt:
		return compareNumericValues(t.Value, taint.Value, t.Operator)
	default:
		return false
	}
}

// compareNumericValues performs numeric comparison between toleration and taint values
func compareNumericValues(tolerationVal, taintVal string, op TolerationOperator) bool {
	tVal, err := strconv.ParseInt(tolerationVal, 10, 64)
	if err != nil {
		return false
	}

	tnVal, err := strconv.ParseInt(taintVal, 10, 64)
	if err != nil {
		return false
	}

	switch op {
	case TolerationOpLt:
		return tVal < tnVal
	case TolerationOpGt:
		return tVal > tnVal
	default:
		return false
	}
}
