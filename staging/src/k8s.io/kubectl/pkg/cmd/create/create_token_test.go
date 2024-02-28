/*
Copyright 2022 The Kubernetes Authors.

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

package create

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/pointer"
	kjson "sigs.k8s.io/json"

	authenticationv1 "k8s.io/api/authentication/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCreateToken(t *testing.T) {
	tests := []struct {
		test string

		name            string
		namespace       string
		output          string
		boundObjectKind string
		boundObjectName string
		boundObjectUID  string
		audiences       []string
		duration        time.Duration

		enableNodeBindingFeature bool

		serverResponseToken string
		serverResponseError string

		expectRequestPath  string
		expectTokenRequest *authenticationv1.TokenRequest

		expectStdout string
		expectStderr string
	}{
		{
			test: "simple",
			name: "mysa",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},

		{
			test:      "custom namespace",
			name:      "custom-sa",
			namespace: "custom-ns",

			expectRequestPath: "/api/v1/namespaces/custom-ns/serviceaccounts/custom-sa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},

		{
			test:   "yaml",
			name:   "mysa",
			output: "yaml",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
			},
			serverResponseToken: "abc",
			expectStdout: `apiVersion: authentication.k8s.io/v1
kind: TokenRequest
metadata:
  creationTimestamp: null
spec:
  audiences: null
  boundObjectRef: null
  expirationSeconds: null
status:
  expirationTimestamp: null
  token: abc
`,
		},

		{
			test:            "bad bound object kind",
			name:            "mysa",
			boundObjectKind: "Foo",
			expectStderr:    `error: supported --bound-object-kind values are Pod, Secret`,
		},
		{
			test:                     "bad bound object kind (node feature enabled)",
			name:                     "mysa",
			enableNodeBindingFeature: true,
			boundObjectKind:          "Foo",
			expectStderr:             `error: supported --bound-object-kind values are Node, Pod, Secret`,
		},
		{
			test:            "missing bound object name",
			name:            "mysa",
			boundObjectKind: "Pod",
			expectStderr:    `error: --bound-object-name is required if --bound-object-kind is provided`,
		},
		{
			test:            "invalid bound object name",
			name:            "mysa",
			boundObjectName: "mypod",
			expectStderr:    `error: --bound-object-name can only be set if --bound-object-kind is provided`,
		},
		{
			test:           "invalid bound object uid",
			name:           "mysa",
			boundObjectUID: "myuid",
			expectStderr:   `error: --bound-object-uid can only be set if --bound-object-kind is provided`,
		},
		{
			test: "valid bound object",
			name: "mysa",

			boundObjectKind: "Pod",
			boundObjectName: "mypod",
			boundObjectUID:  "myuid",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
				Spec: authenticationv1.TokenRequestSpec{
					BoundObjectRef: &authenticationv1.BoundObjectReference{
						Kind:       "Pod",
						APIVersion: "v1",
						Name:       "mypod",
						UID:        "myuid",
					},
				},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},
		{
			test: "valid bound object (Node)",
			name: "mysa",

			enableNodeBindingFeature: true,
			boundObjectKind:          "Node",
			boundObjectName:          "mynode",
			boundObjectUID:           "myuid",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
				Spec: authenticationv1.TokenRequestSpec{
					BoundObjectRef: &authenticationv1.BoundObjectReference{
						Kind:       "Node",
						APIVersion: "v1",
						Name:       "mynode",
						UID:        "myuid",
					},
				},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},
		{
			test:         "invalid audience",
			name:         "mysa",
			audiences:    []string{"test", "", "test2"},
			expectStderr: `error: --audience must not be an empty string`,
		},
		{
			test: "valid audiences",
			name: "mysa",

			audiences: []string{"test,value1", "test,value2"},

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
				Spec: authenticationv1.TokenRequestSpec{
					Audiences: []string{"test,value1", "test,value2"},
				},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},

		{
			test:         "invalid duration",
			name:         "mysa",
			duration:     -1,
			expectStderr: `error: --duration must be greater than or equal to 0`,
		},
		{
			test:         "invalid duration unit",
			name:         "mysa",
			duration:     time.Microsecond,
			expectStderr: `error: --duration cannot be expressed in units less than seconds`,
		},
		{
			test: "valid duration",
			name: "mysa",

			duration: 1000 * time.Second,

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
				Spec: authenticationv1.TokenRequestSpec{
					ExpirationSeconds: pointer.Int64(1000),
				},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},
		{
			test: "zero duration act as default",
			name: "mysa",

			duration: 0 * time.Second,

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
				Spec: authenticationv1.TokenRequestSpec{
					ExpirationSeconds: nil,
				},
			},
			serverResponseToken: "abc",
			expectStdout:        "abc",
		},
		{
			test: "server error",
			name: "mysa",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
			},
			serverResponseError: "bad bad request",
			expectStderr:        `error: failed to create token:  "bad bad request" is invalid`,
		},
		{
			test: "server missing token",
			name: "mysa",

			expectRequestPath: "/api/v1/namespaces/test/serviceaccounts/mysa/token",
			expectTokenRequest: &authenticationv1.TokenRequest{
				TypeMeta: metav1.TypeMeta{APIVersion: "authentication.k8s.io/v1", Kind: "TokenRequest"},
			},
			serverResponseToken: "",
			expectStderr:        `error: failed to create token: no token in server response`,
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			defer cmdutil.DefaultBehaviorOnFatal()
			sawError := ""
			cmdutil.BehaviorOnFatal(func(str string, code int) {
				sawError = str
			})

			namespace := "test"
			if test.namespace != "" {
				namespace = test.namespace
			}
			tf := cmdtesting.NewTestFactory().WithNamespace(namespace)
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{}

			var code int
			var body []byte
			if len(test.serverResponseError) > 0 {
				code = 422
				response := apierrors.NewInvalid(schema.GroupKind{Group: "", Kind: ""}, test.serverResponseError, nil)
				response.ErrStatus.APIVersion = "v1"
				response.ErrStatus.Kind = "Status"
				body, _ = json.Marshal(response.ErrStatus)
			} else {
				code = 200
				response := authenticationv1.TokenRequest{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "authentication.k8s.io/v1",
						Kind:       "TokenRequest",
					},
					Status: authenticationv1.TokenRequestStatus{Token: test.serverResponseToken},
				}
				body, _ = json.Marshal(response)
			}

			ns := scheme.Codecs.WithoutConversion()
			var tokenRequest *authenticationv1.TokenRequest
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					if req.URL.Path != test.expectRequestPath {
						t.Fatalf("expected %q, got %q", test.expectRequestPath, req.URL.Path)
					}
					data, err := io.ReadAll(req.Body)
					if err != nil {
						t.Fatal(err)
					}
					tokenRequest = &authenticationv1.TokenRequest{}
					if strictErrs, err := kjson.UnmarshalStrict(data, tokenRequest); err != nil {
						t.Fatal(err)
					} else if len(strictErrs) > 0 {
						t.Fatal(strictErrs)
					}

					return &http.Response{
						StatusCode: code,
						Body:       io.NopCloser(bytes.NewBuffer(body)),
					}, nil
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, stdout, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdCreateToken(tf, ioStreams)
			if test.output != "" {
				cmd.Flags().Set("output", test.output)
			}
			if test.boundObjectKind != "" {
				cmd.Flags().Set("bound-object-kind", test.boundObjectKind)
			}
			if test.boundObjectName != "" {
				cmd.Flags().Set("bound-object-name", test.boundObjectName)
			}
			if test.boundObjectUID != "" {
				cmd.Flags().Set("bound-object-uid", test.boundObjectUID)
			}
			for _, aud := range test.audiences {
				cmd.Flags().Set("audience", aud)
			}
			if test.duration != 0 {
				cmd.Flags().Set("duration", test.duration.String())
			}
			if test.enableNodeBindingFeature {
				os.Setenv("KUBECTL_NODE_BOUND_TOKENS", "true")
				defer os.Unsetenv("KUBECTL_NODE_BOUND_TOKENS")
			}
			cmd.Run(cmd, []string{test.name})

			if !reflect.DeepEqual(tokenRequest, test.expectTokenRequest) {
				t.Fatalf("unexpected request:\n%s", cmp.Diff(test.expectTokenRequest, tokenRequest))
			}

			if stdout.String() != test.expectStdout {
				t.Errorf("unexpected stdout:\n%s", cmp.Diff(test.expectStdout, stdout.String()))
			}
			if sawError != test.expectStderr {
				t.Errorf("unexpected stderr:\n%s", cmp.Diff(test.expectStderr, sawError))
			}
		})
	}
}
