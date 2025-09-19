// untested sections: 3

package matchers

import (
	"fmt"
	"os"

	"github.com/onsi/gomega/format"
)

type BeAnExistingFileMatcher struct {
	expected any
}

func (matcher *BeAnExistingFileMatcher) Match(actual any) (success bool, err error) {
	actualFilename, ok := actual.(string)
	if !ok {
		return false, fmt.Errorf("BeAnExistingFileMatcher matcher expects a file path")
	}

	if _, err = os.Stat(actualFilename); err != nil {
		switch {
		case os.IsNotExist(err):
			return false, nil
		default:
			return false, err
		}
	}

	return true, nil
}

func (matcher *BeAnExistingFileMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to exist")
}

func (matcher *BeAnExistingFileMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to exist")
}
