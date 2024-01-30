// untested sections: 5

package matchers

import (
	"fmt"
	"os"

	"github.com/onsi/gomega/format"
)

type notADirectoryError struct {
	os.FileInfo
}

func (t notADirectoryError) Error() string {
	fileInfo := os.FileInfo(t)
	switch {
	case fileInfo.Mode().IsRegular():
		return "file is a regular file"
	default:
		return fmt.Sprintf("file mode is: %s", fileInfo.Mode().String())
	}
}

type BeADirectoryMatcher struct {
	expected interface{}
	err      error
}

func (matcher *BeADirectoryMatcher) Match(actual interface{}) (success bool, err error) {
	actualFilename, ok := actual.(string)
	if !ok {
		return false, fmt.Errorf("BeADirectoryMatcher matcher expects a file path")
	}

	fileInfo, err := os.Stat(actualFilename)
	if err != nil {
		matcher.err = err
		return false, nil
	}

	if !fileInfo.Mode().IsDir() {
		matcher.err = notADirectoryError{fileInfo}
		return false, nil
	}
	return true, nil
}

func (matcher *BeADirectoryMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, fmt.Sprintf("to be a directory: %s", matcher.err))
}

func (matcher *BeADirectoryMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not be a directory")
}
