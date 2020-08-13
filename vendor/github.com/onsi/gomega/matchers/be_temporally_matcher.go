// untested sections: 3

package matchers

import (
	"fmt"
	"time"

	"github.com/onsi/gomega/format"
)

type BeTemporallyMatcher struct {
	Comparator string
	CompareTo  time.Time
	Threshold  []time.Duration
}

func (matcher *BeTemporallyMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, fmt.Sprintf("to be %s", matcher.Comparator), matcher.CompareTo)
}

func (matcher *BeTemporallyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, fmt.Sprintf("not to be %s", matcher.Comparator), matcher.CompareTo)
}

func (matcher *BeTemporallyMatcher) Match(actual interface{}) (bool, error) {
	// predicate to test for time.Time type
	isTime := func(t interface{}) bool {
		_, ok := t.(time.Time)
		return ok
	}

	if !isTime(actual) {
		return false, fmt.Errorf("Expected a time.Time.  Got:\n%s", format.Object(actual, 1))
	}

	switch matcher.Comparator {
	case "==", "~", ">", ">=", "<", "<=":
	default:
		return false, fmt.Errorf("Unknown comparator: %s", matcher.Comparator)
	}

	var threshold = time.Millisecond
	if len(matcher.Threshold) == 1 {
		threshold = matcher.Threshold[0]
	}

	return matcher.matchTimes(actual.(time.Time), matcher.CompareTo, threshold), nil
}

func (matcher *BeTemporallyMatcher) matchTimes(actual, compareTo time.Time, threshold time.Duration) (success bool) {
	switch matcher.Comparator {
	case "==":
		return actual.Equal(compareTo)
	case "~":
		diff := actual.Sub(compareTo)
		return -threshold <= diff && diff <= threshold
	case ">":
		return actual.After(compareTo)
	case ">=":
		return !actual.Before(compareTo)
	case "<":
		return actual.Before(compareTo)
	case "<=":
		return !actual.After(compareTo)
	}
	return false
}
