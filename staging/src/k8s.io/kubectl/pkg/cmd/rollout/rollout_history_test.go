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

package rollout

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
)

type fakeHistoryViewer struct {
	viewHistoryFn func(namespace, name string, revision int64) (string, error)
	getHistoryFn  func(namespace, name string) (map[int64]runtime.Object, error)
}

func (h *fakeHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	return h.viewHistoryFn(namespace, name, revision)
}

func (h *fakeHistoryViewer) GetHistory(namespace, name string) (map[int64]runtime.Object, error) {
	return h.getHistoryFn(namespace, name)
}

func setupFakeHistoryViewer(t *testing.T) *fakeHistoryViewer {
	fhv := &fakeHistoryViewer{
		viewHistoryFn: func(namespace, name string, revision int64) (string, error) {
			t.Fatalf("ViewHistory mock not implemented")
			return "", nil
		},
		getHistoryFn: func(namespace, name string) (map[int64]runtime.Object, error) {
			t.Fatalf("GetHistory mock not implemented")
			return nil, nil
		},
	}
	polymorphichelpers.HistoryViewerFn = func(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (polymorphichelpers.HistoryViewer, error) {
		return fhv, nil
	}
	return fhv
}

func TestRolloutHistory(t *testing.T) {
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutPauseGroupVersionEncoder)

	tf.Client = &RolloutPauseRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutPauseGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/foo" && m == "GET":
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = "foo"
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	testCases := map[string]struct {
		flags            map[string]string
		expectedOutput   string
		expectedRevision int64
	}{
		"should display ViewHistory output for all revisions": {
			expectedOutput: `deployment.apps/foo 
Fake ViewHistory Output

`,
			expectedRevision: int64(0),
		},
		"should display ViewHistory output for a single revision": {
			flags: map[string]string{"revision": "2"},
			expectedOutput: `deployment.apps/foo with revision #2
Fake ViewHistory Output

`,
			expectedRevision: int64(2),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(tt *testing.T) {
			fhv := setupFakeHistoryViewer(tt)
			var actualNamespace, actualName *string
			var actualRevision *int64
			fhv.viewHistoryFn = func(namespace, name string, revision int64) (string, error) {
				actualNamespace = &namespace
				actualName = &name
				actualRevision = &revision
				return "Fake ViewHistory Output\n", nil
			}

			streams, _, buf, errBuf := genericiooptions.NewTestIOStreams()
			cmd := NewCmdRolloutHistory(tf, streams)
			for k, v := range tc.flags {
				cmd.Flags().Set(k, v)
			}
			cmd.Run(cmd, []string{"deployment/foo"})

			expectedErrorOutput := ""
			if errBuf.String() != expectedErrorOutput {
				tt.Fatalf("expected error output: %s, but got %s", expectedErrorOutput, errBuf.String())
			}

			if buf.String() != tc.expectedOutput {
				tt.Fatalf("expected output: %s, but got: %s", tc.expectedOutput, buf.String())
			}

			expectedNamespace := "test"
			if actualNamespace == nil || *actualNamespace != expectedNamespace {
				tt.Fatalf("expected ViewHistory to have been called with namespace %s, but it was %v", expectedNamespace, *actualNamespace)
			}

			expectedName := "foo"
			if actualName == nil || *actualName != expectedName {
				tt.Fatalf("expected ViewHistory to have been called with name %s, but it was %v", expectedName, *actualName)
			}

			if actualRevision == nil {
				tt.Fatalf("expected ViewHistory to have been called with revision %d, but it was ", tc.expectedRevision)
			} else if *actualRevision != tc.expectedRevision {
				tt.Fatalf("expected ViewHistory to have been called with revision %d, but it was %v", tc.expectedRevision, *actualRevision)
			}
		})
	}
}

func TestMultipleResourceRolloutHistory(t *testing.T) {
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutPauseGroupVersionEncoder)

	tf.Client = &RolloutPauseRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutPauseGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/foo" && m == "GET":
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = "foo"
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				case p == "/namespaces/test/deployments/bar" && m == "GET":
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = "bar"
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	testCases := map[string]struct {
		flags          map[string]string
		expectedOutput string
	}{
		"should display ViewHistory output for all revisions": {
			expectedOutput: `deployment.apps/foo 
Fake ViewHistory Output

deployment.apps/bar 
Fake ViewHistory Output

`,
		},
		"should display ViewHistory output for a single revision": {
			flags: map[string]string{"revision": "2"},
			expectedOutput: `deployment.apps/foo with revision #2
Fake ViewHistory Output

deployment.apps/bar with revision #2
Fake ViewHistory Output

`,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(tt *testing.T) {
			fhv := setupFakeHistoryViewer(tt)
			fhv.viewHistoryFn = func(namespace, name string, revision int64) (string, error) {
				return "Fake ViewHistory Output\n", nil
			}

			streams, _, buf, errBuf := genericiooptions.NewTestIOStreams()
			cmd := NewCmdRolloutHistory(tf, streams)
			for k, v := range tc.flags {
				cmd.Flags().Set(k, v)
			}
			cmd.Run(cmd, []string{"deployment/foo", "deployment/bar"})

			expectedErrorOutput := ""
			if errBuf.String() != expectedErrorOutput {
				tt.Fatalf("expected error output: %s, but got %s", expectedErrorOutput, errBuf.String())
			}

			if buf.String() != tc.expectedOutput {
				tt.Fatalf("expected output: %s, but got: %s", tc.expectedOutput, buf.String())
			}
		})
	}
}

func TestRolloutHistoryWithOutput(t *testing.T) {
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutPauseGroupVersionEncoder)

	tf.Client = &RolloutPauseRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutPauseGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/foo" && m == "GET":
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = "foo"
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	testCases := map[string]struct {
		flags          map[string]string
		expectedOutput string
	}{
		"json": {
			flags: map[string]string{"revision": "2", "output": "json"},
			expectedOutput: `{
    "kind": "ReplicaSet",
    "apiVersion": "apps/v1",
    "metadata": {
        "name": "rev2"
    },
    "spec": {
        "selector": null,
        "template": {
            "metadata": {},
            "spec": {
                "containers": null
            }
        }
    },
    "status": {
        "replicas": 0
    }
}
`,
		},
		"yaml": {
			flags: map[string]string{"revision": "2", "output": "yaml"},
			expectedOutput: `apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: rev2
spec:
  selector: null
  template:
    metadata: {}
    spec:
      containers: null
status:
  replicas: 0
`,
		},
		"yaml all revisions": {
			flags: map[string]string{"output": "yaml"},
			expectedOutput: `apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: rev1
spec:
  selector: null
  template:
    metadata: {}
    spec:
      containers: null
status:
  replicas: 0
---
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: rev2
spec:
  selector: null
  template:
    metadata: {}
    spec:
      containers: null
status:
  replicas: 0
`,
		},
		"name": {
			flags: map[string]string{"output": "name"},
			expectedOutput: `replicaset.apps/rev1
replicaset.apps/rev2
`,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			fhv := setupFakeHistoryViewer(t)
			var actualNamespace, actualName *string
			fhv.getHistoryFn = func(namespace, name string) (map[int64]runtime.Object, error) {
				actualNamespace = &namespace
				actualName = &name
				return map[int64]runtime.Object{
					1: &appsv1.ReplicaSet{ObjectMeta: v1.ObjectMeta{Name: "rev1"}},
					2: &appsv1.ReplicaSet{ObjectMeta: v1.ObjectMeta{Name: "rev2"}},
				}, nil
			}

			streams, _, buf, errBuf := genericiooptions.NewTestIOStreams()
			cmd := NewCmdRolloutHistory(tf, streams)
			for k, v := range tc.flags {
				cmd.Flags().Set(k, v)
			}
			cmd.Run(cmd, []string{"deployment/foo"})

			expectedErrorOutput := ""
			if errBuf.String() != expectedErrorOutput {
				t.Fatalf("expected error output: %s, but got %s", expectedErrorOutput, errBuf.String())
			}

			if buf.String() != tc.expectedOutput {
				t.Fatalf("expected output: %s, but got: %s", tc.expectedOutput, buf.String())
			}

			expectedNamespace := "test"
			if actualNamespace == nil || *actualNamespace != expectedNamespace {
				t.Fatalf("expected GetHistory to have been called with namespace %s, but it was %v", expectedNamespace, *actualNamespace)
			}

			expectedName := "foo"
			if actualName == nil || *actualName != expectedName {
				t.Fatalf("expected GetHistory to have been called with name %s, but it was %v", expectedName, *actualName)
			}
		})
	}
}

func TestRolloutHistoryErrors(t *testing.T) {
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutPauseGroupVersionEncoder)

	tf.Client = &RolloutPauseRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutPauseGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/foo" && m == "GET":
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = "foo"
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	testCases := map[string]struct {
		revision      int64
		outputFormat  string
		expectedError string
	}{
		"get non-existing revision as yaml": {
			revision:      999,
			outputFormat:  "yaml",
			expectedError: "unable to find the specified revision",
		},
		"get non-existing revision as json": {
			revision:      999,
			outputFormat:  "json",
			expectedError: "unable to find the specified revision",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			fhv := setupFakeHistoryViewer(t)
			fhv.getHistoryFn = func(namespace, name string) (map[int64]runtime.Object, error) {
				return map[int64]runtime.Object{
					1: &appsv1.ReplicaSet{ObjectMeta: v1.ObjectMeta{Name: "rev1"}},
					2: &appsv1.ReplicaSet{ObjectMeta: v1.ObjectMeta{Name: "rev2"}},
				}, nil
			}

			streams := genericiooptions.NewTestIOStreamsDiscard()
			o := NewRolloutHistoryOptions(streams)

			printFlags := &genericclioptions.PrintFlags{
				JSONYamlPrintFlags: &genericclioptions.JSONYamlPrintFlags{
					ShowManagedFields: true,
				},
				OutputFormat: &tc.outputFormat,
				OutputFlagSpecified: func() bool {
					return true
				},
			}

			o.PrintFlags = printFlags
			o.Revision = tc.revision

			if err := o.Complete(tf, nil, []string{"deployment/foo"}); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			err := o.Run()
			if err != nil && err.Error() != tc.expectedError {
				t.Fatalf("expected '%s' error, but got: %v", tc.expectedError, err)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	opts := RolloutHistoryOptions{
		Revision:  0,
		Resources: []string{"deployment/foo"},
	}
	if err := opts.Validate(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	}

	opts.Revision = -1
	expectedError := "revision must be a positive integer: -1"
	if err := opts.Validate(); err == nil {
		t.Fatalf("unexpected non error")
	} else if err.Error() != expectedError {
		t.Fatalf("expected error %s, but got %s", expectedError, err.Error())
	}

	opts.Revision = 0
	opts.Resources = []string{}
	expectedError = "required resource not specified"
	if err := opts.Validate(); err == nil {
		t.Fatalf("unexpected non error")
	} else if err.Error() != expectedError {
		t.Fatalf("expected error %s, but got %s", expectedError, err.Error())
	}
}
