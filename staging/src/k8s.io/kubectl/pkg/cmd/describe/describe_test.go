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

package describe

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/describe"
	"k8s.io/kubectl/pkg/scheme"
)

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestDescribeUnknownSchemaObject(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	tf := cmdtesting.NewTestFactory().WithNamespace("non-default")
	defer tf.Cleanup()
	_, _, codec := cmdtesting.NewExternalScheme()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, cmdtesting.NewInternalType("", "", "foo"))},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"type", "foo"})

	if d.Name != "foo" || d.Namespace != "" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != d.Output {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestDescribeUnknownNamespacedSchemaObject(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	_, _, codec := cmdtesting.NewExternalScheme()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, cmdtesting.NewInternalNamespacedType("", "", "foo", "non-default"))},
	}
	tf.WithNamespace("non-default")

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"namespacedtype", "foo"})

	if d.Name != "foo" || d.Namespace != "non-default" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != d.Output {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeObject(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	_, _, rc := cmdtesting.TestData()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Run(cmd, []string{})

	if d.Name != "redis-master" || d.Namespace != "test" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != d.Output {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeListObjects(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods, _, _ := cmdtesting.TestData()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods"})
	if buf.String() != fmt.Sprintf("%s\n\n%s", d.Output, d.Output) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeObjectShowEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods, _, _ := cmdtesting.TestData()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	cmd := NewCmdDescribe("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
	cmd.Flags().Set("show-events", "true")
	cmd.Run(cmd, []string{"pods"})
	if d.Settings.ShowEvents != true {
		t.Errorf("ShowEvents = true expected, got ShowEvents = %v", d.Settings.ShowEvents)
	}
}

func TestDescribeObjectSkipEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods, _, _ := cmdtesting.TestData()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	cmd := NewCmdDescribe("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
	cmd.Flags().Set("show-events", "false")
	cmd.Run(cmd, []string{"pods"})
	if d.Settings.ShowEvents != false {
		t.Errorf("ShowEvents = false expected, got ShowEvents = %v", d.Settings.ShowEvents)
	}
}

func TestDescribeObjectChunkSize(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods, _, _ := cmdtesting.TestData()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	cmd := NewCmdDescribe("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
	cmd.Flags().Set("chunk-size", "100")
	cmd.Run(cmd, []string{"pods"})
	if d.Settings.ChunkSize != 100 {
		t.Errorf("ChunkSize = 100 expected, got ChunkSize = %v", d.Settings.ChunkSize)
	}
}

func TestDescribeHelpMessage(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.SetArgs([]string{"-h"})
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	_, err := cmd.ExecuteC()

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	got := buf.String()

	expected := `describe (-f FILENAME | TYPE [NAME_PREFIX | -l label] | TYPE/NAME)`
	if !strings.Contains(got, expected) {
		t.Errorf("Expected to contain: \n %v\nGot:\n %v\n", expected, got)
	}

	unexpected := `describe (-f FILENAME | TYPE [NAME_PREFIX | -l label] | TYPE/NAME) [flags]`
	if strings.Contains(got, unexpected) {
		t.Errorf("Expected not to contain: \n %v\nGot:\n %v\n", unexpected, got)
	}
}

func TestDescribeNoResourcesFound(t *testing.T) {
	testNS := "testns"
	testCases := []struct {
		name           string
		flags          map[string]string
		namespace      string
		expectedOutput string
		expectedErr    string
	}{
		{
			name:           "all namespaces",
			flags:          map[string]string{"all-namespaces": "true"},
			expectedOutput: "",
			expectedErr:    "No resources found\n",
		},
		{
			name:           "all in namespace",
			namespace:      testNS,
			expectedOutput: "",
			expectedErr:    "No resources found in " + testNS + " namespace.\n",
		},
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			pods, _, _ := cmdtesting.EmptyTestData()
			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()
			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
			}

			streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

			cmd := NewCmdDescribe("kubectl", tf, streams)
			for name, value := range testCase.flags {
				_ = cmd.Flags().Set(name, value)
			}
			cmd.Run(cmd, []string{"pods"})

			if e, a := testCase.expectedOutput, buf.String(); e != a {
				t.Errorf("Unexpected output:\nExpected:\n%v\nActual:\n%v", e, a)
			}
			if e, a := testCase.expectedErr, errbuf.String(); e != a {
				t.Errorf("Unexpected error:\nExpected:\n%v\nActual:\n%v", e, a)
			}
		})
	}
}

// TestDescribeShowEventsCardinality tests the cardinality-based ShowEvents
// default for the standard list and direct-name paths.
func TestDescribeShowEventsCardinality(t *testing.T) {
	onePod := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}
	twoPods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}

	testCases := []struct {
		name             string
		args             []string
		flags            map[string]string
		pods             *corev1.PodList
		exactName        string // if set, use direct GET client instead of list Resp
		expectShowEvents bool
	}{
		{
			name:             "single object keeps events",
			args:             []string{"pods"},
			pods:             onePod,
			expectShowEvents: true,
		},
		{
			name:             "multiple objects defaults events off",
			args:             []string{"pods"},
			pods:             twoPods,
			expectShowEvents: false,
		},
		{
			name:             "multiple objects explicit --show-events=true",
			args:             []string{"pods"},
			pods:             twoPods,
			flags:            map[string]string{"show-events": "true"},
			expectShowEvents: true,
		},
		{
			name:             "multiple objects explicit --show-events=false",
			args:             []string{"pods"},
			pods:             twoPods,
			flags:            map[string]string{"show-events": "false"},
			expectShowEvents: false,
		},
		{
			name:             "single object by exact name keeps events",
			args:             []string{"pods", "my-pod"},
			exactName:        "my-pod",
			expectShowEvents: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			d := &testDescriber{Output: "test output"}
			oldFn := describe.DescriberFn
			defer func() { describe.DescriberFn = oldFn }()
			describe.DescriberFn = d.describerFor

			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()
			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			if tc.exactName != "" {
				pod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: tc.exactName, Namespace: "test", ResourceVersion: "10"},
					Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
				}
				tf.UnstructuredClient = &fake.RESTClient{
					NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
					Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
						if req.URL.Path == "/namespaces/test/pods/"+tc.exactName && req.Method == http.MethodGet {
							return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pod)}, nil
						}
						t.Fatalf("unexpected request: %s %s", req.Method, req.URL.Path)
						return nil, nil
					}),
				}
			} else {
				tf.UnstructuredClient = &fake.RESTClient{
					NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
					Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, tc.pods)},
				}
			}

			cmd := NewCmdDescribe("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
			for k, v := range tc.flags {
				_ = cmd.Flags().Set(k, v)
			}
			cmd.Run(cmd, tc.args)

			if d.Settings.ShowEvents != tc.expectShowEvents {
				t.Errorf("expected ShowEvents=%v, got %v", tc.expectShowEvents, d.Settings.ShowEvents)
			}
		})
	}
}

// TestDescribeShowEventsPrefixPath tests the cardinality-based ShowEvents
// default for the prefix-match path (kubectl describe TYPE NAME_PREFIX).
func TestDescribeShowEventsPrefixPath(t *testing.T) {
	twoFooOneBar := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar-xyz", Namespace: "test", ResourceVersion: "12"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}
	twoPodsFoo := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}
	oneFooOneBar := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar-xyz", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}
	barOnly := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "bar-abc", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}

	testCases := []struct {
		name             string
		pods             *corev1.PodList
		flags            map[string]string
		expectShowEvents bool
		expectMatchCount int // 0 when expecting error
		expectFatal      bool
	}{
		{
			name:             "multi match defaults events off",
			pods:             twoFooOneBar,
			expectShowEvents: false,
			expectMatchCount: 2,
		},
		{
			name:             "single match still disables events (prefix path)",
			pods:             oneFooOneBar,
			expectShowEvents: false,
			expectMatchCount: 1,
		},
		{
			name:             "multi match explicit --show-events=true",
			pods:             twoPodsFoo,
			flags:            map[string]string{"show-events": "true"},
			expectShowEvents: true,
			expectMatchCount: 2,
		},
		{
			name:             "multi match explicit --show-events=false",
			pods:             twoPodsFoo,
			flags:            map[string]string{"show-events": "false"},
			expectShowEvents: false,
			expectMatchCount: 2,
		},
		{
			name:        "no match returns original error",
			pods:        barOnly,
			expectFatal: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			d := &testDescriber{Output: "test output"}
			oldFn := describe.DescriberFn
			defer func() { describe.DescriberFn = oldFn }()
			describe.DescriberFn = d.describerFor

			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()
			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == "/namespaces/test/pods/foo" && m == "GET":
						return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
					case p == "/namespaces/test/pods" && m == "GET":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, tc.pods)}, nil
					default:
						return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
					}
				}),
			}

			var fatalMsg string
			cmdutil.BehaviorOnFatal(func(msg string, code int) { fatalMsg = msg })
			defer cmdutil.DefaultBehaviorOnFatal()

			streams, _, buf, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdDescribe("kubectl", tf, streams)
			for k, v := range tc.flags {
				_ = cmd.Flags().Set(k, v)
			}
			cmd.Run(cmd, []string{"pods", "foo"})

			if tc.expectFatal {
				if buf.Len() != 0 {
					t.Errorf("expected no output, got: %q", buf.String())
				}
				if fatalMsg == "" {
					t.Error("expected fatal error, got none")
				}
				return
			}
			count := strings.Count(buf.String(), d.Output)
			if count != tc.expectMatchCount {
				t.Errorf("expected %d describe outputs, got %d", tc.expectMatchCount, count)
			}
			if d.Settings.ShowEvents != tc.expectShowEvents {
				t.Errorf("expected ShowEvents=%v, got %v", tc.expectShowEvents, d.Settings.ShowEvents)
			}
		})
	}
}

// TestDescribePartialErrorStillDisablesEvents verifies that when ContinueOnError
// produces partial results, the cardinality check applies to successful infos.
func TestDescribePartialErrorStillDisablesEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() { describe.DescriberFn = oldFn }()
	describe.DescriberFn = d.describerFor

	foo := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}}
	baz := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods/foo" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, foo)}, nil
			case p == "/namespaces/test/pods/bar" && m == "GET":
				return &http.Response{StatusCode: http.StatusInternalServerError, Header: cmdtesting.DefaultHeader(),
					Body: cmdtesting.StringBody(`{"kind":"Status","apiVersion":"v1","status":"Failure","message":"internal error"}`)}, nil
			case p == "/namespaces/test/pods/baz" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, baz)}, nil
			default:
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			}
		}),
	}

	var fatalMsg string
	cmdutil.BehaviorOnFatal(func(msg string, code int) { fatalMsg = msg })
	defer cmdutil.DefaultBehaviorOnFatal()

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo", "bar", "baz"})

	if count := strings.Count(buf.String(), d.Output); count != 2 {
		t.Errorf("expected 2 describe outputs for partial success, got %d", count)
	}
	if d.Settings.ShowEvents != false {
		t.Errorf("expected ShowEvents=false for 2 successful infos, got %v", d.Settings.ShowEvents)
	}
	if fatalMsg == "" {
		t.Error("expected partial error to be reported")
	}
}

// TestDescribeMixedResourceTypesDisablesEvents verifies that the cardinality
// check applies across resource types (1 pod + 1 service = events off).
func TestDescribeMixedResourceTypesDisablesEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() { describe.DescriberFn = oldFn }()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items:    []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "my-pod", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}}},
	}
	services := &corev1.ServiceList{
		ListMeta: metav1.ListMeta{ResourceVersion: "16"},
		Items:    []corev1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "my-svc", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.ServiceSpec{Type: corev1.ServiceTypeClusterIP}}},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case p == "/namespaces/test/services" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, services)}, nil
			default:
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			}
		}),
	}

	cmd := NewCmdDescribe("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
	cmd.Run(cmd, []string{"pods,services"})
	if d.Settings.ShowEvents != false {
		t.Errorf("expected ShowEvents=false for mixed types (2 objects), got %v", d.Settings.ShowEvents)
	}
}

// TestDescribeDescriberErrorsRetainDisabledEvents verifies that when 3 objects
// are fetched (events disabled) but the describer fails for 2, the successful
// describe still receives ShowEvents=false.
func TestDescribeDescriberErrorsRetainDisabledEvents(t *testing.T) {
	cd := &conditionalDescriber{
		Output:    "test output",
		FailNames: map[string]bool{"foo": true, "bar": true},
	}
	oldFn := describe.DescriberFn
	defer func() { describe.DescriberFn = oldFn }()
	describe.DescriberFn = cd.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
			{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"}, Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways}},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	var fatalMsg string
	cmdutil.BehaviorOnFatal(func(msg string, code int) { fatalMsg = msg })
	defer cmdutil.DefaultBehaviorOnFatal()

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods"})

	if !strings.Contains(buf.String(), cd.Output) {
		t.Errorf("expected baz's output in result, got: %q", buf.String())
	}
	if cd.LastSettings.ShowEvents != false {
		t.Errorf("expected ShowEvents=false for 3 infos, got %v", cd.LastSettings.ShowEvents)
	}
	if fatalMsg == "" {
		t.Error("expected describer errors to be reported")
	}
}

type testDescriber struct {
	Name, Namespace string
	Settings        describe.DescriberSettings
	Output          string
	Err             error
}

func (t *testDescriber) Describe(namespace, name string, describerSettings describe.DescriberSettings) (output string, err error) {
	t.Namespace, t.Name = namespace, name
	t.Settings = describerSettings
	return t.Output, t.Err
}
func (t *testDescriber) describerFor(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (describe.ResourceDescriber, error) {
	return t, nil
}

// conditionalDescriber fails Describe() for names in FailNames, succeeds for others.
type conditionalDescriber struct {
	FailNames    map[string]bool
	Output       string
	LastSettings describe.DescriberSettings
}

func (c *conditionalDescriber) Describe(namespace, name string, settings describe.DescriberSettings) (string, error) {
	c.LastSettings = settings
	if c.FailNames[name] {
		return "", fmt.Errorf("describe failed for %s", name)
	}
	return c.Output, nil
}

func (c *conditionalDescriber) describerFor(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (describe.ResourceDescriber, error) {
	return c, nil
}
