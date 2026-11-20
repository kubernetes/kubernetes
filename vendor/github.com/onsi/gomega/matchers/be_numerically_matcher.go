// untested sections: 4

package matchers

import (
	"fmt"
	"math"
	"math/big"

	"github.com/onsi/gomega/format"
)

type BeNumericallyMatcher struct {
	Comparator string
	CompareTo  []any
}

func (matcher *BeNumericallyMatcher) FailureMessage(actual any) (message string) {
	return matcher.FormatFailureMessage(actual, false)
}

func (matcher *BeNumericallyMatcher) NegatedFailureMessage(actual any) (message string) {
	return matcher.FormatFailureMessage(actual, true)
}

func (matcher *BeNumericallyMatcher) FormatFailureMessage(actual any, negated bool) (message string) {
	if len(matcher.CompareTo) == 1 {
		message = fmt.Sprintf("to be %s", matcher.Comparator)
	} else {
		message = fmt.Sprintf("to be within %v of %s", matcher.CompareTo[1], matcher.Comparator)
	}
	if negated {
		message = "not " + message
	}
	return format.Message(actual, message, matcher.CompareTo[0])
}

func (matcher *BeNumericallyMatcher) Match(actual any) (success bool, err error) {
	if len(matcher.CompareTo) == 0 || len(matcher.CompareTo) > 2 {
		return false, fmt.Errorf("BeNumerically requires 1 or 2 CompareTo arguments.  Got:\n%s", format.Object(matcher.CompareTo, 1))
	}
	if !isNumber(actual) {
		return false, fmt.Errorf("Expected a number.  Got:\n%s", format.Object(actual, 1))
	}
	if !isNumber(matcher.CompareTo[0]) {
		return false, fmt.Errorf("Expected a number.  Got:\n%s", format.Object(matcher.CompareTo[0], 1))
	}
	if len(matcher.CompareTo) == 2 && !isNumber(matcher.CompareTo[1]) {
		return false, fmt.Errorf("Expected a number.  Got:\n%s", format.Object(matcher.CompareTo[1], 1))
	}

	switch matcher.Comparator {
	case "==", "~", ">", ">=", "<", "<=":
	default:
		return false, fmt.Errorf("Unknown comparator: %s", matcher.Comparator)
	}

	if isFloat(actual) || isFloat(matcher.CompareTo[0]) {
		var secondOperand float64 = 1e-8
		if len(matcher.CompareTo) == 2 {
			secondOperand = toFloat(matcher.CompareTo[1])
		}
		success = matcher.matchFloats(toFloat(actual), toFloat(matcher.CompareTo[0]), secondOperand)
	} else if isInteger(actual) || isUnsignedInteger(actual) {
		var threshold any = 0
		if len(matcher.CompareTo) == 2 {
			threshold = matcher.CompareTo[1]
		}
		success = matcher.matchIntegers(toBigInt(actual), toBigInt(matcher.CompareTo[0]), threshold)
	} else {
		return false, fmt.Errorf("Failed to compare:\n%s\n%s:\n%s", format.Object(actual, 1), matcher.Comparator, format.Object(matcher.CompareTo[0], 1))
	}

	return success, nil
}

// matchIntegers compares signed and unsigned integers by their exact mathematical value, using math/big so
// that neither mixing signedness nor computing the distance between the two values can overflow
func (matcher *BeNumericallyMatcher) matchIntegers(actual, compareTo *big.Int, threshold any) (success bool) {
	switch matcher.Comparator {
	case "==", "~":
		distance := new(big.Int).Sub(actual, compareTo)
		return isWithinThreshold(distance.Abs(distance), threshold)
	case ">":
		return actual.Cmp(compareTo) > 0
	case ">=":
		return actual.Cmp(compareTo) >= 0
	case "<":
		return actual.Cmp(compareTo) < 0
	case "<=":
		return actual.Cmp(compareTo) <= 0
	}
	return false
}

// isWithinThreshold reports whether the (non-negative) distance is no greater than the threshold, which may be any number
func isWithinThreshold(distance *big.Int, threshold any) bool {
	if isFloat(threshold) {
		t := toFloat(threshold)
		if math.IsNaN(t) {
			return false
		}
		return new(big.Float).SetInt(distance).Cmp(big.NewFloat(t)) <= 0
	}
	return distance.Cmp(toBigInt(threshold)) <= 0
}

func (matcher *BeNumericallyMatcher) matchFloats(actual, compareTo, threshold float64) (success bool) {
	switch matcher.Comparator {
	case "~":
		return math.Abs(actual-compareTo) <= threshold
	case "==":
		// an explicit threshold is honored, as it is for integers; without one == means exact equality
		if len(matcher.CompareTo) == 2 {
			return math.Abs(actual-compareTo) <= threshold
		}
		return (actual == compareTo)
	case ">":
		return (actual > compareTo)
	case ">=":
		return (actual >= compareTo)
	case "<":
		return (actual < compareTo)
	case "<=":
		return (actual <= compareTo)
	}
	return false
}
