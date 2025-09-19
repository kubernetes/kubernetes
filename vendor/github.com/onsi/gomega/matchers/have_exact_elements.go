package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type mismatchFailure struct {
	failure string
	index   int
}

type HaveExactElementsMatcher struct {
	Elements         []any
	mismatchFailures []mismatchFailure
	missingIndex     int
	extraIndex       int
}

func (matcher *HaveExactElementsMatcher) Match(actual any) (success bool, err error) {
	matcher.resetState()

	if isMap(actual) || miter.IsSeq2(actual) {
		return false, fmt.Errorf("HaveExactElements matcher doesn't work on map or iter.Seq2.  Got:\n%s", format.Object(actual, 1))
	}

	matchers := matchers(matcher.Elements)
	lenMatchers := len(matchers)

	success = true

	if miter.IsIter(actual) {
		// In the worst case, we need to see everything before we can give our
		// verdict. The only exception is fast fail.
		i := 0
		miter.IterateV(actual, func(v reflect.Value) bool {
			if i >= lenMatchers {
				// the iterator produces more values than we got matchers: this
				// is not good.
				matcher.extraIndex = i
				success = false
				return false
			}

			elemMatcher := matchers[i].(omegaMatcher)
			match, err := elemMatcher.Match(v.Interface())
			if err != nil {
				matcher.mismatchFailures = append(matcher.mismatchFailures, mismatchFailure{
					index:   i,
					failure: err.Error(),
				})
				success = false
			} else if !match {
				matcher.mismatchFailures = append(matcher.mismatchFailures, mismatchFailure{
					index:   i,
					failure: elemMatcher.FailureMessage(v.Interface()),
				})
				success = false
			}
			i++
			return true
		})
		if i < len(matchers) {
			// the iterator produced less values than we got matchers: this is
			// no good, no no no.
			matcher.missingIndex = i
			success = false
		}
		return success, nil
	}

	values := valuesOf(actual)
	lenValues := len(values)

	for i := 0; i < lenMatchers || i < lenValues; i++ {
		if i >= lenMatchers {
			matcher.extraIndex = i
			success = false
			continue
		}

		if i >= lenValues {
			matcher.missingIndex = i
			success = false
			return
		}

		elemMatcher := matchers[i].(omegaMatcher)
		match, err := elemMatcher.Match(values[i])
		if err != nil {
			matcher.mismatchFailures = append(matcher.mismatchFailures, mismatchFailure{
				index:   i,
				failure: err.Error(),
			})
			success = false
		} else if !match {
			matcher.mismatchFailures = append(matcher.mismatchFailures, mismatchFailure{
				index:   i,
				failure: elemMatcher.FailureMessage(values[i]),
			})
			success = false
		}
	}

	return success, nil
}

func (matcher *HaveExactElementsMatcher) FailureMessage(actual any) (message string) {
	message = format.Message(actual, "to have exact elements with", presentable(matcher.Elements))
	if matcher.missingIndex > 0 {
		message = fmt.Sprintf("%s\nthe missing elements start from index %d", message, matcher.missingIndex)
	}
	if matcher.extraIndex > 0 {
		message = fmt.Sprintf("%s\nthe extra elements start from index %d", message, matcher.extraIndex)
	}
	if len(matcher.mismatchFailures) != 0 {
		message = fmt.Sprintf("%s\nthe mismatch indexes were:", message)
	}
	for _, mismatch := range matcher.mismatchFailures {
		message = fmt.Sprintf("%s\n%d: %s", message, mismatch.index, mismatch.failure)
	}
	return
}

func (matcher *HaveExactElementsMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to contain elements", presentable(matcher.Elements))
}

func (matcher *HaveExactElementsMatcher) resetState() {
	matcher.mismatchFailures = nil
	matcher.missingIndex = 0
	matcher.extraIndex = 0
}
