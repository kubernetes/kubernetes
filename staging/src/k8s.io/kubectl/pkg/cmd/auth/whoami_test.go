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

package auth

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/printers"
	authfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestWhoAmIRun(t *testing.T) {
	tests := []struct {
		name      string
		o         *WhoAmIOptions
		args      []string
		serverErr error

		expectedError       error
		expectedBodyStrings []string
	}{
		{
			name: "success test",
			o: &WhoAmIOptions{
				resourcePrinterFunc: printTableSelfSubjectAccessReview,
			},
			args: []string{},
			expectedBodyStrings: []string{
				`ATTRIBUTE         VALUE`,
				`Username          jane.doe`,
				`UID               uniq-id`,
				`Groups            [students teachers]`,
				`Extra: skills     [reading learning]`,
				`Extra: subjects   [math sports]`,
				``,
			},
		},
		{
			name: "JSON test",
			o: &WhoAmIOptions{
				resourcePrinterFunc: printers.NewTypeSetter(scheme.Scheme).ToPrinter(&printers.JSONPrinter{}).PrintObj,
			},
			args: []string{},
			expectedBodyStrings: []string{
				`{
    "kind": "SelfSubjectReview",
    "apiVersion": "authentication.k8s.io/v1alpha1",
    "metadata": {
        "creationTimestamp": null
    },
    "status": {
        "userInfo": {
            "username": "jane.doe",
            "uid": "uniq-id",
            "groups": [
                "students",
                "teachers"
            ],
            "extra": {
                "skills": [
                    "reading",
                    "learning"
                ],
                "subjects": [
                    "math",
                    "sports"
                ]
            }
        }
    }
}
`,
			},
		},
		{
			name: "Forbidden error",
			o: &WhoAmIOptions{
				resourcePrinterFunc: printTableSelfSubjectAccessReview,
			},
			args: []string{},
			serverErr: errors.NewForbidden(
				corev1.Resource("selfsubjectreviews"), "foo", fmt.Errorf("error"),
			),
			expectedError:       forbiddenErr,
			expectedBodyStrings: []string{},
		},
		{
			name: "NotFound error",
			o: &WhoAmIOptions{
				resourcePrinterFunc: printTableSelfSubjectAccessReview,
			},
			args:                []string{},
			serverErr:           errors.NewNotFound(corev1.Resource("selfsubjectreviews"), "foo"),
			expectedError:       notEnabledErr,
			expectedBodyStrings: []string{},
		},
		{
			name: "Server error",
			o: &WhoAmIOptions{
				resourcePrinterFunc: printTableSelfSubjectAccessReview,
			},
			args:                []string{},
			serverErr:           fmt.Errorf("a random server-side error"),
			expectedError:       fmt.Errorf("a random server-side error"),
			expectedBodyStrings: []string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var b bytes.Buffer

			test.o.Out = &b
			test.o.ErrOut = io.Discard

			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			fakeAuthClientSet := &authfake.Clientset{}

			fakeAuthClientSet.AddReactor("create", "selfsubjectreviews",
				func(action core.Action) (handled bool, ret runtime.Object, err error) {
					if test.serverErr != nil {
						return true, nil, test.serverErr
					}

					res := &authenticationv1alpha1.SelfSubjectReview{
						Status: authenticationv1alpha1.SelfSubjectReviewStatus{
							UserInfo: authenticationv1.UserInfo{
								Username: "jane.doe",
								UID:      "uniq-id",
								Groups:   []string{"students", "teachers"},
								Extra: map[string]authenticationv1.ExtraValue{
									"subjects": {"math", "sports"},
									"skills":   {"reading", "learning"},
								},
							},
						},
					}
					return true, res, nil
				})
			test.o.authClient = fakeAuthClientSet.AuthenticationV1alpha1()

			err := test.o.Run()
			switch {
			case test.expectedError == nil && err == nil:
				// pass
			case err != nil && test.expectedError != nil && strings.Contains(err.Error(), test.expectedError.Error()):
				// pass
			default:
				t.Errorf("%s: expected %v, got %v", test.name, test.expectedError, err)
				return
			}

			res := b.String()
			expectedBody := strings.Join(test.expectedBodyStrings, "\n")

			if expectedBody != res {
				t.Errorf("%s: expected \n%q, got \n%q", test.name, expectedBody, res)
			}
		})
	}
}
