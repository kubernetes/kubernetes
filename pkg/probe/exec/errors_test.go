/*
Copyright 2021 The Kubernetes Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
