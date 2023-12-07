/*
Copyright 2023 The Kubernetes Authors.

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

func TestErrors(t *testing.T) {
	tests := []struct {
		err     error
		timeout time.Duration
		message string
	}{
		{
			err:     errors.New("some error message"),
			timeout: time.Hour * 8,
			message: "some error message",
		},
	}

	for i, test := range tests {
		testErr := NewTimeoutError(test.err, test.timeout)

		if testErr == nil {
			t.Errorf("[%d] expected error a TimeoutError, got nil", i)
		}
		if msg := testErr.Error(); msg != test.message {
			t.Errorf("[%d] expected error message %q, got %q", i, test.message, msg)
		}
		if timeout := testErr.Timeout(); timeout != test.timeout {
			t.Errorf("[%d] expected timeout %q, got %q", i, test.timeout, timeout)
		}
	}
}
