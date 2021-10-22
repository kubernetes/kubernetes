// +build go1.7

package management_test

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"fmt"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// TestIsResourceNotFoundError tests IsResourceNotFoundError with the
// set of given test cases.
func TestIsResourceNotFoundError(t *testing.T) {
	// isResourceNotFoundTestCases is a set of structs comprising of the error
	// IsResourceNotFoundError should test and the expected result.
	var isResourceNotFoundTestCases = []struct {
		err      error
		expected bool
	}{
		{nil, false},
		{fmt.Errorf("Some other random error."), false},
		{management.AzureError{Code: "ResourceNotFound"}, true},
		{management.AzureError{Code: "NotAResourceNotFound"}, false},
	}

	for i, testCase := range isResourceNotFoundTestCases {
		if res := management.IsResourceNotFoundError(testCase.err); res != testCase.expected {
			t.Fatalf("Test %d: error %s - expected %t - got %t", i+1, testCase.err, testCase.expected, res)
		}
	}
}
