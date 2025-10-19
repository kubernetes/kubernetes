// untested sections: 5

package matchers

import (
	"fmt"
	"os"

	"github.com/onsi/gomega/format"
)

type notARegularFileError struct {
	os.FileInfo
}

func (t notARegularFileError) Error() string {
	fileInfo := os.FileInfo(t)
	switch {
	case fileInfo.IsDir():
		return "file is a directory"
	default:
		return fmt.Sprintf("file mode is: %s", fileInfo.Mode().String())
	}
}

type BeARegularFileMatcher struct {
	expected any
	err      error
}

func (matcher *BeARegularFileMatcher) Match(actual any) (success bool, err error) {
	actualFilename, ok := actual.(string)
	if !ok {
		return false, fmt.Errorf("BeARegularFileMatcher matcher expects a file path")
	}

	fileInfo, err := os.Stat(actualFilename)
	if err != nil {
		matcher.err = err
		return false, nil
	}

	if !fileInfo.Mode().IsRegular() {
		matcher.err = notARegularFileError{fileInfo}
		return false, nil
	}
	return true, nil
}

func (matcher *BeARegularFileMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, fmt.Sprintf("to be a regular file: %s", matcher.err))
}

func (matcher *BeARegularFileMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not be a regular file")
}
