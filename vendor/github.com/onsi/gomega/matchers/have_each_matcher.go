package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type HaveEachMatcher struct {
	Element any
}

func (matcher *HaveEachMatcher) Match(actual any) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) && !miter.IsIter(actual) {
		return false, fmt.Errorf("HaveEach matcher expects an array/slice/map/iter.Seq/iter.Seq2.  Got:\n%s",
			format.Object(actual, 1))
	}

	elemMatcher, elementIsMatcher := matcher.Element.(omegaMatcher)
	if !elementIsMatcher {
		elemMatcher = &EqualMatcher{Expected: matcher.Element}
	}

	if miter.IsIter(actual) {
		// rejecting the non-elements case works different for iterators as we
		// don't want to fetch all elements into a slice first.
		count := 0
		var success bool
		var err error
		if miter.IsSeq2(actual) {
			miter.IterateKV(actual, func(k, v reflect.Value) bool {
				count++
				success, err = elemMatcher.Match(v.Interface())
				if err != nil {
					return false
				}
				return success
			})
		} else {
			miter.IterateV(actual, func(v reflect.Value) bool {
				count++
				success, err = elemMatcher.Match(v.Interface())
				if err != nil {
					return false
				}
				return success
			})
		}
		if count == 0 {
			return false, fmt.Errorf("HaveEach matcher expects a non-empty iter.Seq/iter.Seq2.  Got:\n%s",
				format.Object(actual, 1))
		}
		return success, err
	}

	value := reflect.ValueOf(actual)
	if value.Len() == 0 {
		return false, fmt.Errorf("HaveEach matcher expects a non-empty array/slice/map.  Got:\n%s",
			format.Object(actual, 1))
	}

	var valueAt func(int) any
	if isMap(actual) {
		keys := value.MapKeys()
		valueAt = func(i int) any {
			return value.MapIndex(keys[i]).Interface()
		}
	} else {
		valueAt = func(i int) any {
			return value.Index(i).Interface()
		}
	}

	// if we never failed then we succeed; the empty/nil cases have already been
	// rejected above.
	for i := 0; i < value.Len(); i++ {
		success, err := elemMatcher.Match(valueAt(i))
		if err != nil {
			return false, err
		}
		if !success {
			return false, nil
		}
	}

	return true, nil
}

// FailureMessage returns a suitable failure message.
func (matcher *HaveEachMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to contain element matching", matcher.Element)
}

// NegatedFailureMessage returns a suitable negated failure message.
func (matcher *HaveEachMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to contain element matching", matcher.Element)
}
