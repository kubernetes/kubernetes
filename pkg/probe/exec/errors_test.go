package exec

import (
	"errors"
	"testing"
	"time"
)

func TestNewTimeoutError(t *testing.T) {
	testErr := NewTimeoutError(nil, 0)
	if testErr.err != nil || testErr.timeout != 0 {
		t.Errorf("NewTimeoutError(nil, 0) != testErr")
	}
}

func TestError(t *testing.T) {
	errText := "some example error text"
	testErr := NewTimeoutError(errors.New(errText), 0)
	if testErr.Error() != errText {
		t.Errorf("testErr.Error() != \"%s\"", errText)
	}
}

func TestTimeout(t *testing.T) {
	testTimeout := time.Duration(1)
	testErr := NewTimeoutError(nil, testTimeout)
	if testErr.Timeout() != testTimeout {
		t.Errorf("testErr.Timeout() != %d", testTimeout)
	}
}
