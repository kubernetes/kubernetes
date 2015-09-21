/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"fmt"
	"testing"
)

func TestErrorList(t *testing.T) {
	errList := ErrorList{}
	err := errList.ToError()
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if a := errorListInternal(errList).Error(); a != "" {
		t.Errorf("expected empty string, got %q", a)
	}

	testCases := []struct {
		errs     ErrorList
		expected string
	}{
		{ErrorList{fmt.Errorf("abc")}, "abc"},
		{ErrorList{fmt.Errorf("abc"), fmt.Errorf("123")}, "[abc, 123]"},
	}
	for _, testCase := range testCases {
		err := testCase.errs.ToError()
		if err == nil {
			t.Errorf("expected an error, got nil: %v", testCase)
			continue
		}
		if err.Error() != testCase.expected {
			t.Errorf("expected %q, got %q", testCase.expected, err.Error())
		}
	}
}
