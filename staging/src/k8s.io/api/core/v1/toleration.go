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
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/validate/content"
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
//  5. If enableComparisonOperators is false and the toleration uses 'Lt' or 'Gt'
//     operators, the toleration does not match (returns false).
func (t *Toleration) ToleratesTaint(taint *Taint, enableComparisonOperators bool) (bool, error) {
	if len(t.Effect) > 0 && t.Effect != taint.Effect {
		return false, nil
	}

	if len(t.Key) > 0 && t.Key != taint.Key {
		return false, nil
	}

	// TODO: Use proper defaulting when Toleration becomes a field of PodSpec
	switch t.Operator {
	// empty operator means Equal
	case "", TolerationOpEqual:
		return t.Value == taint.Value, nil
	case TolerationOpExists:
		return true, nil
	case TolerationOpLt, TolerationOpGt:
		// If comparison operators are disabled, this toleration doesn't match
		if !enableComparisonOperators {
			return false, nil
		}
		return compareNumericValues(t.Value, taint.Value, t.Operator)
	default:
		return false, nil
	}
}

// compareNumericValues performs numeric comparison between toleration and taint values
func compareNumericValues(tolerationVal, taintVal string, op TolerationOperator) (bool, error) {
	errorMsgs := content.IsDecimalInteger(tolerationVal)
	if len(errorMsgs) > 0 {
		return false, fmt.Errorf("failed to parse toleration value %q as int64: %s", tolerationVal, strings.Join(errorMsgs, ","))
	}
	tVal, err := strconv.ParseInt(tolerationVal, 10, 64)
	if err != nil {
		return false, fmt.Errorf("failed to parse toleration value %q as int64: %w", tolerationVal, err)
	}

	errorMsgs = content.IsDecimalInteger(taintVal)
	if len(errorMsgs) > 0 {
		return false, fmt.Errorf("failed to parse taint value %q as int64: %s", taintVal, strings.Join(errorMsgs, ","))
	}
	tntVal, err := strconv.ParseInt(taintVal, 10, 64)
	if err != nil {
		return false, fmt.Errorf("failed to parse taint value %q as int64: %w", taintVal, err)
	}

	switch op {
	case TolerationOpLt:
		return tntVal < tVal, nil
	case TolerationOpGt:
		return tntVal > tVal, nil
	default:
		return false, nil
	}
}
