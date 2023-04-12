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

package annotate

import (
	"bytes"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestValidateAnnotationOverwrites(t *testing.T) {
	tests := []struct {
		meta        *metav1.ObjectMeta
		annotations map[string]string
		expectErr   bool
		scenario    string
	}{
		{
			meta: &metav1.ObjectMeta{
				Annotations: map[string]string{
					"a": "A",
					"b": "B",
				},
			},
			annotations: map[string]string{
				"a": "a",
				"c": "C",
			},
			scenario:  "share first annotation",
			expectErr: true,
		},
		{
			meta: &metav1.ObjectMeta{
				Annotations: map[string]string{
					"a": "A",
					"c": "C",
				},
			},
			annotations: map[string]string{
				"b": "B",
				"c": "c",
			},
			scenario:  "share second annotation",
			expectErr: true,
		},
		{
			meta: &metav1.ObjectMeta{
				Annotations: map[string]string{
					"a": "A",
					"c": "C",
				},
			},
			annotations: map[string]string{
				"b": "B",
				"d": "D",
			},
			scenario: "no overlap",
		},
		{
			meta: &metav1.ObjectMeta{},
			annotations: map[string]string{
				"a": "A",
				"b": "B",
			},
			scenario: "no annotations",
		},
	}
	for _, test := range tests {
		err := validateNoAnnotationOverwrites(test.meta, test.annotations)
		if test.expectErr && err == nil {
			t.Errorf("%s: unexpected non-error", test.scenario)
		} else if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", test.scenario, err)
		}
	}
}

func TestParseAnnotations(t *testing.T) {
	testURL := "https://test.com/index.htm?id=123#u=user-name"
	testJSON := `'{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"ReplicationController","namespace":"default","name":"my-nginx","uid":"c544ee78-2665-11e5-8051-42010af0c213","apiVersion":"v1","resourceVersion":"61368"}}'`
	tests := []struct {
		annotations    []string
		expected       map[string]string
		expectedRemove []string
		scenario       string
		expectedErr    string
		expectErr      bool
	}{
		{
			annotations:    []string{"a=b", "c=d"},
			expected:       map[string]string{"a": "b", "c": "d"},
			expectedRemove: []string{},
			scenario:       "add two annotations",
			expectErr:      false,
		},
		{
			annotations:    []string{"url=" + testURL, "fake.kubernetes.io/annotation=" + testJSON},
			expected:       map[string]string{"url": testURL, "fake.kubernetes.io/annotation": testJSON},
			expectedRemove: []string{},
			scenario:       "add annotations with special characters",
			expectErr:      false,
		},
		{
			annotations:    []string{},
			expected:       map[string]string{},
			expectedRemove: []string{},
			scenario:       "add no annotations",
			expectErr:      false,
		},
		{
			annotations:    []string{"a=b", "c=d", "e-"},
			expected:       map[string]string{"a": "b", "c": "d"},
			expectedRemove: []string{"e"},
			scenario:       "add two annotations, remove one",
			expectErr:      false,
		},
		{
			annotations: []string{"ab", "c=d"},
			expectedErr: "invalid annotation format: ab",
			scenario:    "incorrect annotation input (missing =value)",
			expectErr:   true,
		},
		{
			annotations:    []string{"a="},
			expected:       map[string]string{"a": ""},
			expectedRemove: []string{},
			scenario:       "add valid annotation with empty value",
			expectErr:      false,
		},
		{
			annotations: []string{"ab", "a="},
			expectedErr: "invalid annotation format: ab",
			scenario:    "incorrect annotation input (missing =value)",
			expectErr:   true,
		},
		{
			annotations: []string{"-"},
			expectedErr: "invalid annotation format: -",
			scenario:    "incorrect annotation input (missing key)",
			expectErr:   true,
		},
		{
			annotations: []string{"=bar"},
			expectedErr: "invalid annotation format: =bar",
			scenario:    "incorrect annotation input (missing key)",
			expectErr:   true,
		},
	}
	for _, test := range tests {
		annotations, remove, err := parseAnnotations(test.annotations)
		switch {
		case test.expectErr && err == nil:
			t.Errorf("%s: unexpected non-error, should return %v", test.scenario, test.expectedErr)
		case test.expectErr && err.Error() != test.expectedErr:
			t.Errorf("%s: unexpected error %v, expected %v", test.scenario, err, test.expectedErr)
		case !test.expectErr && err != nil:
			t.Errorf("%s: unexpected error %v", test.scenario, err)
		case !test.expectErr && !reflect.DeepEqual(annotations, test.expected):
			t.Errorf("%s: expected %v, got %v", test.scenario, test.expected, annotations)
		case !test.expectErr && !reflect.DeepEqual(remove, test.expectedRemove):
			t.Errorf("%s: expected %v, got %v", test.scenario, test.expectedRemove, remove)
		}
	}
}

func TestValidateAnnotations(t *testing.T) {
	tests := []struct {
		removeAnnotations []string
		newAnnotations    map[string]string
		expectedErr       string
		scenario          string
	}{
		{
			expectedErr:       "can not both modify and remove the following annotation(s) in the same command: a",
			removeAnnotations: []string{"a"},
			newAnnotations:    map[string]string{"a": "b", "c": "d"},
			scenario:          "remove an added annotation",
		},
		{
			expectedErr:       "can not both modify and remove the following annotation(s) in the same command: a, c",
			removeAnnotations: []string{"a", "c"},
			newAnnotations:    map[string]string{"a": "b", "c": "d"},
			scenario:          "remove added annotations",
		},
	}
	for _, test := range tests {
		if err := validateAnnotations(test.removeAnnotations, test.newAnnotations); err == nil {
			t.Errorf("%s: unexpected non-error", test.scenario)
		} else if err.Error() != test.expectedErr {
			t.Errorf("%s: expected error %s, got %s", test.scenario, test.expectedErr, err.Error())
		}
	}
}

func TestUpdateAnnotations(t *testing.T) {
	tests := []struct {
		obj         runtime.Object
		overwrite   bool
		version     string
		annotations map[string]string
		remove      []string
		expected    runtime.Object
		expectedErr string
	}{
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{"a": "b"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{"a": "c"},
			expectedErr: "--overwrite is false but found the following declared annotation(s): 'a' already has a value (b)",
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{"a": "c"},
			overwrite:   true,
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "c"},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{"c": "d"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b", "c": "d"},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{"c": "d"},
			version:     "2",
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations:     map[string]string{"a": "b", "c": "d"},
					ResourceVersion: "2",
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
			annotations: map[string]string{},
			remove:      []string{"a"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b", "c": "d"},
				},
			},
			annotations: map[string]string{"e": "f"},
			remove:      []string{"a"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"c": "d",
						"e": "f",
					},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b", "c": "d"},
				},
			},
			annotations: map[string]string{"e": "f"},
			remove:      []string{"g"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"a": "b",
						"c": "d",
						"e": "f",
					},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b", "c": "d"},
				},
			},
			remove: []string{"e"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"a": "b",
						"c": "d",
					},
				},
			},
		},
		{
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			annotations: map[string]string{"a": "b"},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"a": "b"},
				},
			},
		},
	}
	for _, test := range tests {
		options := &AnnotateOptions{
			overwrite:         test.overwrite,
			newAnnotations:    test.annotations,
			removeAnnotations: test.remove,
			resourceVersion:   test.version,
		}
		err := options.updateAnnotations(test.obj)
		if test.expectedErr != "" {
			if err == nil {
				t.Errorf("unexpected non-error: %v", test)
			}
			if err.Error() != test.expectedErr {
				t.Errorf("error expected: %v, got: %v", test.expectedErr, err.Error())
			}
			continue
		}
		if test.expectedErr == "" && err != nil {
			t.Errorf("unexpected error: %v %v", err, test)
		}
		if !reflect.DeepEqual(test.obj, test.expected) {
			t.Errorf("expected: %v, got %v", test.expected, test.obj)
		}
	}
}

func TestAnnotateErrors(t *testing.T) {
	testCases := map[string]struct {
		args  []string
		errFn func(error) bool
	}{
		"no args": {
			args:  []string{},
			errFn: func(err error) bool { return strings.Contains(err.Error(), "one or more resources must be specified") },
		},
		"not enough annotations": {
			args: []string{"pods"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), "at least one annotation update is required")
			},
		},
		"wrong annotations": {
			args: []string{"pods", "-"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), "at least one annotation update is required")
			},
		},
		"wrong annotations 2": {
			args: []string{"pods", "=bar"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), "at least one annotation update is required")
			},
		},
		"no resources remove annotations": {
			args:  []string{"pods-"},
			errFn: func(err error) bool { return strings.Contains(err.Error(), "one or more resources must be specified") },
		},
		"no resources add annotations": {
			args:  []string{"pods=bar"},
			errFn: func(err error) bool { return strings.Contains(err.Error(), "one or more resources must be specified") },
		},
	}

	for k, testCase := range testCases {
		t.Run(k, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			iostreams, _, bufOut, bufErr := genericiooptions.NewTestIOStreams()
			cmd := NewCmdAnnotate("kubectl", tf, iostreams)
			cmd.SetOut(bufOut)
			cmd.SetErr(bufOut)

			flags := NewAnnotateFlags(iostreams)
			_, err := flags.ToOptions(tf, cmd, testCase.args)
			if !testCase.errFn(err) {
				t.Errorf("%s: unexpected error: %v", k, err)
				return
			}
			if bufOut.Len() > 0 {
				t.Errorf("buffer should be empty: %s", bufOut.String())
			}
			if bufErr.Len() > 0 {
				t.Errorf("buffer should be empty: %s", bufErr.String())
			}
		})
	}
}

func TestAnnotateObject(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "testgroup", Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.Method {
			case "GET":
				switch req.URL.Path {
				case "/namespaces/test/pods/foo":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			case "PATCH":
				switch req.URL.Path {
				case "/namespaces/test/pods/foo":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			default:
				t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	iostreams, _, bufOut, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAnnotate("kubectl", tf, iostreams)
	cmd.SetOut(bufOut)
	cmd.SetErr(bufOut)
	flags := NewAnnotateFlags(iostreams)
	args := []string{"pods/foo", "a=b", "c-"}

	options, err := flags.ToOptions(tf, cmd, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := options.RunAnnotate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAnnotateResourceVersion(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "testgroup", Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.Method {
			case "GET":
				switch req.URL.Path {
				case "/namespaces/test/pods/foo":
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     cmdtesting.DefaultHeader(),
						Body: io.NopCloser(bytes.NewBufferString(
							`{"kind":"Pod","apiVersion":"v1","metadata":{"name":"foo","namespace":"test","resourceVersion":"10"}}`,
						))}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			case "PATCH":
				switch req.URL.Path {
				case "/namespaces/test/pods/foo":
					body, err := io.ReadAll(req.Body)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(body, []byte(`{"metadata":{"annotations":{"a":"b"},"resourceVersion":"10"}}`)) {
						t.Fatalf("expected patch with resourceVersion set, got %s", string(body))
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     cmdtesting.DefaultHeader(),
						Body: io.NopCloser(bytes.NewBufferString(
							`{"kind":"Pod","apiVersion":"v1","metadata":{"name":"foo","namespace":"test","resourceVersion":"11"}}`,
						))}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			default:
				t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	iostreams, _, bufOut, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAnnotate("kubectl", tf, iostreams)
	cmd.SetOut(bufOut)
	cmd.SetErr(bufOut)
	//options := NewAnnotateOptions(iostreams)
	flags := NewAnnotateFlags(iostreams)
	flags.resourceVersion = "10"
	args := []string{"pods/foo", "a=b"}

	options, err := flags.ToOptions(tf, cmd, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := options.RunAnnotate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAnnotateObjectFromFile(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "testgroup", Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.Method {
			case "GET":
				switch req.URL.Path {
				case "/namespaces/test/replicationcontrollers/cassandra":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			case "PATCH":
				switch req.URL.Path {
				case "/namespaces/test/replicationcontrollers/cassandra":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			default:
				t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	iostreams, _, bufOut, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAnnotate("kubectl", tf, iostreams)
	cmd.SetOut(bufOut)
	cmd.SetErr(bufOut)
	flags := NewAnnotateFlags(iostreams)
	flags.Filenames = []string{"../../../testdata/controller.yaml"}
	args := []string{"a=b", "c-"}

	options, err := flags.ToOptions(tf, cmd, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := options.RunAnnotate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAnnotateLocal(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "testgroup", Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	iostreams, _, _, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAnnotate("kubectl", tf, iostreams)
	flags := NewAnnotateFlags(iostreams)
	flags.Local = true
	flags.Filenames = []string{"../../../testdata/controller.yaml"}
	args := []string{"a=b"}

	options, err := flags.ToOptions(tf, cmd, args)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := options.RunAnnotate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAnnotateMultipleObjects(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "testgroup", Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.Method {
			case "GET":
				switch req.URL.Path {
				case "/namespaces/test/pods":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			case "PATCH":
				switch req.URL.Path {
				case "/namespaces/test/pods/foo":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])}, nil
				case "/namespaces/test/pods/bar":
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[1])}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			default:
				t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	iostreams, _, _, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAnnotate("kubectl", tf, iostreams)
	cmd.SetOut(iostreams.Out)
	cmd.SetErr(iostreams.Out)
	flags := NewAnnotateFlags(iostreams)
	flags.All = true
	args := []string{"pods", "a=b", "c-"}

	options, err := flags.ToOptions(tf, cmd, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := options.RunAnnotate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
