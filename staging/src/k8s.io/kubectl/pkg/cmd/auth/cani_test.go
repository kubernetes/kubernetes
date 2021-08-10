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
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestRunAccessCheck(t *testing.T) {
	tests := []struct {
		name      string
		o         *CanIOptions
		args      []string
		allowed   bool
		serverErr error

		expectedBodyStrings []string
	}{
		{
			name:    "restmapping for args",
			o:       &CanIOptions{},
			args:    []string{"get", "replicaset"},
			allowed: true,
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"namespace":"test","verb":"get","group":"extensions","resource":"replicasets"}}`,
			},
		},
		{
			name:    "simple success",
			o:       &CanIOptions{},
			args:    []string{"get", "deployments.extensions/foo"},
			allowed: true,
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"namespace":"test","verb":"get","group":"extensions","resource":"deployments","name":"foo"}}`,
			},
		},
		{
			name: "all namespaces",
			o: &CanIOptions{
				AllNamespaces: true,
			},
			args:    []string{"get", "deployments.extensions/foo"},
			allowed: true,
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"verb":"get","group":"extensions","resource":"deployments","name":"foo"}}`,
			},
		},
		{
			name: "disallowed",
			o: &CanIOptions{
				AllNamespaces: true,
			},
			args:    []string{"get", "deployments.extensions/foo"},
			allowed: false,
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"verb":"get","group":"extensions","resource":"deployments","name":"foo"}}`,
			},
		},
		{
			name: "forcedError",
			o: &CanIOptions{
				AllNamespaces: true,
			},
			args:      []string{"get", "deployments.extensions/foo"},
			allowed:   false,
			serverErr: fmt.Errorf("forcedError"),
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"verb":"get","group":"extensions","resource":"deployments","name":"foo"}}`,
			},
		},
		{
			name: "sub resource",
			o: &CanIOptions{
				AllNamespaces: true,
				Subresource:   "log",
			},
			args:    []string{"get", "pods"},
			allowed: true,
			expectedBodyStrings: []string{
				`{"resourceAttributes":{"verb":"get","resource":"pods","subresource":"log"}}`,
			},
		},
		{
			name:    "nonResourceURL",
			o:       &CanIOptions{},
			args:    []string{"get", "/logs"},
			allowed: true,
			expectedBodyStrings: []string{
				`{"nonResourceAttributes":{"path":"/logs","verb":"get"}}`,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.o.Out = ioutil.Discard
			test.o.ErrOut = ioutil.Discard

			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					expectPath := "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews"
					if req.URL.Path != expectPath {
						t.Errorf("%s: expected %v, got %v", test.name, expectPath, req.URL.Path)
						return nil, nil
					}
					bodyBits, err := ioutil.ReadAll(req.Body)
					if err != nil {
						t.Errorf("%s: %v", test.name, err)
						return nil, nil
					}
					body := string(bodyBits)

					for _, expectedBody := range test.expectedBodyStrings {
						if !strings.Contains(body, expectedBody) {
							t.Errorf("%s expecting %s in %s", test.name, expectedBody, body)
						}
					}

					return &http.Response{
							StatusCode: http.StatusOK,
							Body: ioutil.NopCloser(bytes.NewBufferString(
								fmt.Sprintf(`{"kind":"SelfSubjectAccessReview","apiVersion":"authorization.k8s.io/v1","status":{"allowed":%v}}`, test.allowed),
							)),
						},
						test.serverErr
				}),
			}
			tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}}

			if err := test.o.Complete(tf, test.args); err != nil {
				t.Errorf("%s: %v", test.name, err)
				return
			}

			actualAllowed, err := test.o.RunAccessCheck()
			switch {
			case test.serverErr == nil && err == nil:
				// pass
			case err != nil && test.serverErr != nil && strings.Contains(err.Error(), test.serverErr.Error()):
				// pass
			default:
				t.Errorf("%s: expected %v, got %v", test.name, test.serverErr, err)
				return
			}
			if actualAllowed != test.allowed {
				t.Errorf("%s: expected %v, got %v", test.name, test.allowed, actualAllowed)
				return
			}
		})
	}
}

func TestRunAccessList(t *testing.T) {
	t.Run("test access list", func(t *testing.T) {
		options := &CanIOptions{List: true}
		expectedOutput := "Resources   Non-Resource URLs   Resource Names    Verbs\n" +
			"job.*       []                  [test-resource]   [get list]\n" +
			"pod.*       []                  [test-resource]   [get list]\n" +
			"            [/apis/*]           []                [get]\n" +
			"            [/version]          []                [get]\n"

		tf := cmdtesting.NewTestFactory().WithNamespace("test")
		defer tf.Cleanup()

		ns := scheme.Codecs.WithoutConversion()
		codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

		tf.Client = &fake.RESTClient{
			GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch req.URL.Path {
				case "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
					body := ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, getSelfSubjectRulesReview()))))
					return &http.Response{StatusCode: http.StatusOK, Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
		options.IOStreams = ioStreams
		if err := options.Complete(tf, []string{}); err != nil {
			t.Errorf("got unexpected error when do Complete(): %v", err)
			return
		}

		err := options.RunAccessList()
		if err != nil {
			t.Errorf("got unexpected error when do RunAccessList(): %v", err)
		} else if buf.String() != expectedOutput {
			t.Errorf("expected %v\n but got %v\n", expectedOutput, buf.String())
		}
	})
}

func TestRunResourceFor(t *testing.T) {
	tests := []struct {
		name        string
		o           *CanIOptions
		resourceArg string

		expectGVR      schema.GroupVersionResource
		expectedErrOut string
	}{
		{
			name:        "any resources",
			o:           &CanIOptions{},
			resourceArg: "*",
			expectGVR: schema.GroupVersionResource{
				Resource: "*",
			},
		},
		{
			name:        "server-supported standard resources without group",
			o:           &CanIOptions{},
			resourceArg: "pods",
			expectGVR: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			name:        "server-supported standard resources with group",
			o:           &CanIOptions{},
			resourceArg: "jobs",
			expectGVR: schema.GroupVersionResource{
				Group:    "batch",
				Version:  "v1",
				Resource: "jobs",
			},
		},
		{
			name:        "server-supported nonstandard resources",
			o:           &CanIOptions{},
			resourceArg: "users",
			expectGVR: schema.GroupVersionResource{
				Resource: "users",
			},
		},
		{
			name:        "invalid resources",
			o:           &CanIOptions{},
			resourceArg: "invalid",
			expectGVR: schema.GroupVersionResource{
				Resource: "invalid",
			},
			expectedErrOut: "Warning: the server doesn't have a resource type 'invalid'\n",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			ioStreams, _, _, buf := genericclioptions.NewTestIOStreams()
			test.o.IOStreams = ioStreams

			restMapper, err := tf.ToRESTMapper()
			if err != nil {
				t.Errorf("got unexpected error when do tf.ToRESTMapper(): %v", err)
				return
			}
			gvr := test.o.resourceFor(restMapper, test.resourceArg)
			if gvr != test.expectGVR {
				t.Errorf("expected %v\n but got %v\n", test.expectGVR, gvr)
			}
			if buf.String() != test.expectedErrOut {
				t.Errorf("expected %v\n but got %v\n", test.expectedErrOut, buf.String())
			}
		})
	}
}

func getSelfSubjectRulesReview() *authorizationv1.SelfSubjectRulesReview {
	return &authorizationv1.SelfSubjectRulesReview{
		Status: authorizationv1.SubjectRulesReviewStatus{
			ResourceRules: []authorizationv1.ResourceRule{
				{
					Verbs:         []string{"get", "list"},
					APIGroups:     []string{"*"},
					Resources:     []string{"pod", "job"},
					ResourceNames: []string{"test-resource"},
				},
			},
			NonResourceRules: []authorizationv1.NonResourceRule{
				{
					Verbs:           []string{"get"},
					NonResourceURLs: []string{"/apis/*", "/version"},
				},
			},
		},
	}
}
