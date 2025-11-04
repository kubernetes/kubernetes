// untested sections: 2

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type BeEmptyMatcher struct {
}

func (matcher *BeEmptyMatcher) Match(actual any) (success bool, err error) {
	// short-circuit the iterator case, as we only need to see the first
	// element, if any.
	if miter.IsIter(actual) {
		var length int
		if miter.IsSeq2(actual) {
			miter.IterateKV(actual, func(k, v reflect.Value) bool { length++; return false })
		} else {
			miter.IterateV(actual, func(v reflect.Value) bool { length++; return false })
		}
		return length == 0, nil
	}

	length, ok := lengthOf(actual)
	if !ok {
		return false, fmt.Errorf("BeEmpty matcher expects a string/array/map/channel/slice/iterator.  Got:\n%s", format.Object(actual, 1))
	}

	return length == 0, nil
}

func (matcher *BeEmptyMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be empty")
}

func (matcher *BeEmptyMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be empty")
}
