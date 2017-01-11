/*
Copyright 2014 The Kubernetes Authors.

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

package union

import (
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

type mockAuthzHandler struct {
	isAuthorized bool
	err          error
}

func (mock *mockAuthzHandler) Authorize(a authorizer.Attributes) (bool, string, error) {
	if mock.err != nil {
		return false, "", mock.err
	}
	if !mock.isAuthorized {
		return false, "", nil
	}
	return true, "", nil
}

func TestAuthorization(t *testing.T) {
	tests := []struct {
		firstHandler    *mockAuthzHandler
		secondHandler   *mockAuthzHandler
		expectedFailure bool
		expectedError   bool
	}{
		{
			firstHandler:    &mockAuthzHandler{isAuthorized: false},
			secondHandler:   &mockAuthzHandler{isAuthorized: true},
			expectedFailure: false,
			expectedError:   false,
		},
		{
			firstHandler:    &mockAuthzHandler{isAuthorized: true},
			secondHandler:   &mockAuthzHandler{isAuthorized: false},
			expectedFailure: false,
			expectedError:   false,
		},
		{
			firstHandler:    &mockAuthzHandler{isAuthorized: false},
			secondHandler:   &mockAuthzHandler{isAuthorized: false},
			expectedFailure: true,
			expectedError:   false,
		},
		{
			firstHandler:    &mockAuthzHandler{err: fmt.Errorf("foo")},
			secondHandler:   &mockAuthzHandler{err: fmt.Errorf("foo")},
			expectedFailure: true,
			expectedError:   true,
		},
	}

	for _, tc := range tests {
		authzHandler := New(tc.firstHandler, tc.secondHandler)

		authorized, _, err := authzHandler.Authorize(nil)
		if tc.expectedError && err == nil {
			t.Errorf("Expected error")
		} else if !tc.expectedError && err != nil {
			t.Fatal(err)
		}

		if tc.expectedFailure && authorized {
			t.Errorf("Expected failed authorization")
		} else if !tc.expectedFailure && !authorized {
			t.Errorf("Unexpected authorization failure")
		}
	}
}
