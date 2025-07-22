// untested sections: 4

package matchers

import (
	"fmt"
	"math"

	"github.com/onsi/gomega/format"
)

type BeNumericallyMatcher struct {
	Comparator string
	CompareTo  []interface{}
}

func (matcher *BeNumericallyMatcher) FailureMessage(actual interface{}) (message string) {
	return matcher.FormatFailureMessage(actual, false)
}

func (matcher *BeNumericallyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return matcher.FormatFailureMessage(actual, true)
}

func (matcher *BeNumericallyMatcher) FormatFailureMessage(actual interface{}, negated bool) (message string) {
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

func (matcher *BeNumericallyMatcher) Match(actual interface{}) (success bool, err error) {
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
	} else if isInteger(actual) {
		var secondOperand int64 = 0
		if len(matcher.CompareTo) == 2 {
			secondOperand = toInteger(matcher.CompareTo[1])
		}
		success = matcher.matchIntegers(toInteger(actual), toInteger(matcher.CompareTo[0]), secondOperand)
	} else if isUnsignedInteger(actual) {
		var secondOperand uint64 = 0
		if len(matcher.CompareTo) == 2 {
			secondOperand = toUnsignedInteger(matcher.CompareTo[1])
		}
		success = matcher.matchUnsignedIntegers(toUnsignedInteger(actual), toUnsignedInteger(matcher.CompareTo[0]), secondOperand)
	} else {
		return false, fmt.Errorf("Failed to compare:\n%s\n%s:\n%s", format.Object(actual, 1), matcher.Comparator, format.Object(matcher.CompareTo[0], 1))
	}

	return success, nil
}

func (matcher *BeNumericallyMatcher) matchIntegers(actual, compareTo, threshold int64) (success bool) {
	switch matcher.Comparator {
	case "==", "~":
		diff := actual - compareTo
		return -threshold <= diff && diff <= threshold
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

func (matcher *BeNumericallyMatcher) matchUnsignedIntegers(actual, compareTo, threshold uint64) (success bool) {
	switch matcher.Comparator {
	case "==", "~":
		if actual < compareTo {
			actual, compareTo = compareTo, actual
		}
		return actual-compareTo <= threshold
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

func (matcher *BeNumericallyMatcher) matchFloats(actual, compareTo, threshold float64) (success bool) {
	switch matcher.Comparator {
	case "~":
		return math.Abs(actual-compareTo) <= threshold
	case "==":
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
