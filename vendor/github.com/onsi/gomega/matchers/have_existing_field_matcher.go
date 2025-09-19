package matchers

import (
	"errors"
	"fmt"

	"github.com/onsi/gomega/format"
)

type HaveExistingFieldMatcher struct {
	Field string
}

func (matcher *HaveExistingFieldMatcher) Match(actual any) (success bool, err error) {
	// we don't care about the field's actual value, just about any error in
	// trying to find the field (or method).
	_, err = extractField(actual, matcher.Field, "HaveExistingField")
	if err == nil {
		return true, nil
	}
	var mferr missingFieldError
	if errors.As(err, &mferr) {
		// missing field errors aren't errors in this context, but instead
		// unsuccessful matches.
		return false, nil
	}
	return false, err
}

func (matcher *HaveExistingFieldMatcher) FailureMessage(actual any) (message string) {
	return fmt.Sprintf("Expected\n%s\nto have field '%s'", format.Object(actual, 1), matcher.Field)
}

func (matcher *HaveExistingFieldMatcher) NegatedFailureMessage(actual any) (message string) {
	return fmt.Sprintf("Expected\n%s\nnot to have field '%s'", format.Object(actual, 1), matcher.Field)
}
