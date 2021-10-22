// +build go1.7

package management_test

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

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
