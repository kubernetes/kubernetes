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

package drain

import (
	"context"
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestRunCordonOrUncordon(t *testing.T) {
	nilContextError := fmt.Errorf("RunCordonOrUncordon error: drainer.Ctx can't be nil")
	nilClientError := fmt.Errorf("RunCordonOrUncordon error: drainer.Client can't be nil")
	tests := []struct {
		description   string
		drainer       *Helper
		node          *corev1.Node
		desired       bool
		expectedError *error
	}{
		{
			description: "nil context object",
			drainer: &Helper{
				Client: fake.NewSimpleClientset(),
			},
			desired:       true,
			expectedError: &nilContextError,
		},
		{
			description: "nil client object",
			drainer: &Helper{
				Ctx: context.TODO(),
			},
			desired:       true,
			expectedError: &nilClientError,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.description, func(t *testing.T) {
			err := RunCordonOrUncordon(test.drainer, test.node, test.desired)
			if test.expectedError == nil {
				if err != nil {
					t.Fatalf("%s: did not expect error, got err=%s", test.description, err.Error())
				}
			} else if err.Error() != (*test.expectedError).Error() {
				t.Fatalf("%s: the error does not match expected error, got err=%s, expected err=%s", test.description, err, *test.expectedError)
			}
		})
	}
}
