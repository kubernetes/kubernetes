/*
Copyright 2017 The Kubernetes Authors.

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

package test

import (
	"strings"
	"testing"
)

// ExpectError calls t.Fatalf if the error does not contain a substr match.
// If substr is empty, a nil error is expected.
// It is useful to call ExpectError from subtests.
func ExpectError(t *testing.T, err error, substr string) {
	if err != nil {
		if len(substr) == 0 {
			t.Fatalf("expect nil error but got %q", err.Error())
		} else if !strings.Contains(err.Error(), substr) {
			t.Fatalf("expect error to contain %q but got %q", substr, err.Error())
		}
	} else if len(substr) > 0 {
		t.Fatalf("expect error to contain %q but got nil error", substr)
	}
}

// SkipRest returns true if there was a non-nil error or if we expected an error that didn't happen,
// and logs the appropriate error on the test object.
// The return value indicates whether we should skip the rest of the test case due to the error result.
func SkipRest(t *testing.T, desc string, err error, contains string) bool {
	if err != nil {
		if len(contains) == 0 {
			t.Errorf("case %q, expect nil error but got %q", desc, err.Error())
		} else if !strings.Contains(err.Error(), contains) {
			t.Errorf("case %q, expect error to contain %q but got %q", desc, contains, err.Error())
		}
		return true
	} else if len(contains) > 0 {
		t.Errorf("case %q, expect error to contain %q but got nil error", desc, contains)
		return true
	}
	return false
}
