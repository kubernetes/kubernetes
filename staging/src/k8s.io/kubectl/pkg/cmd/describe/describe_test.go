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

// TestDescribeMatchingResourcesMultipleMatches verifies that when multiple resources
// match a prefix, all matching resources are described in the output.
func TestDescribeMatchingResourcesMultipleMatches(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar-xyz", Namespace: "test", ResourceVersion: "12"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/foo" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		default:
			// Typed client requests (e.g., batch event fetch via KubernetesClientSet)
			// hit /api/v1/ paths. Return empty event list so the batch path works
			// or falls back gracefully to per-resource describing.
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.EventList{})}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"})

	// Both foo-abc and foo-def should be described, but not bar-xyz.
	if !strings.Contains(buf.String(), d.Output) {
		t.Errorf("expected output to contain %q, got: %q", d.Output, buf.String())
	}
	// Output should contain two copies of the describer output (one per matching pod).
	count := strings.Count(buf.String(), d.Output)
	if count != 2 {
		t.Errorf("expected 2 describe outputs for prefix match, got %d", count)
	}
}

// TestDescribeMatchingResourcesSingleMatch verifies that a single prefix match
// uses the normal per-resource event path (no batching needed).
func TestDescribeMatchingResourcesSingleMatch(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar-xyz", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/foo" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		default:
			// Return 404 for any unexpected paths (API discovery, namespace checks, etc.)
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"})

	// Single match uses the fallback path (no batching for 1 match).
	// ShowEvents should be true since batching is not triggered.
	if d.Settings.ShowEvents != true {
		t.Errorf("expected ShowEvents=true for single match, got %v", d.Settings.ShowEvents)
	}
	if buf.String() != fmt.Sprintf("%s\n", d.Output) {
		t.Errorf("unexpected output: %q", buf.String())
	}
}

// TestDescribeMatchingResourcesShortPrefix verifies that a prefix shorter than
// minPrefixLength is rejected and returns the original NotFound error.
func TestDescribeMatchingResourcesShortPrefix(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			// The exact name lookup returns 404, triggering the prefix-match path.
			// But since the prefix is too short, no pod list request should be made.
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		}),
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	// Capture fatal errors without failing the test — the short prefix
	// should cause the original NotFound error to propagate via CheckErr.
	var fatalMsg string
	cmdutil.BehaviorOnFatal(func(msg string, code int) {
		fatalMsg = msg
	})
	defer cmdutil.DefaultBehaviorOnFatal()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "fo"}) // "fo" is 2 chars, below minPrefixLength

	if buf.Len() != 0 {
		t.Errorf("expected no output for short prefix, got: %q", buf.String())
	}
	if fatalMsg == "" {
		t.Error("expected fatal error for short prefix, got none")
	}
}

// TestDescribeMatchingResourcesEventCap verifies that when more than
// maxPrefixResourcesForEvents match, events are skipped and a warning is printed.
func TestDescribeMatchingResourcesEventCap(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	// Create 12 pods all matching prefix "nginx-" to exceed the cap of 10.
	var podItems []corev1.Pod
	for i := 0; i < 12; i++ {
		podItems = append(podItems, corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("nginx-%03d", i),
				Namespace:       "test",
				ResourceVersion: fmt.Sprintf("%d", 10+i),
			},
			Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
		})
	}
	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "100"},
		Items:    podItems,
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/nginx" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		default:
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "nginx"})

	// All 12 pods should be described.
	count := strings.Count(buf.String(), d.Output)
	if count != 12 {
		t.Errorf("expected 12 describe outputs, got %d", count)
	}

	// Events should be skipped (ShowEvents=false passed to describer).
	if d.Settings.ShowEvents != false {
		t.Errorf("expected ShowEvents=false when over cap, got %v", d.Settings.ShowEvents)
	}

	// Warning should be printed to stderr.
	if !strings.Contains(errbuf.String(), "skipping event queries") {
		t.Errorf("expected event cap warning in stderr, got: %q", errbuf.String())
	}
}

// TestDescribeMatchingResourcesBoundaryPrefix verifies that a prefix of exactly
// minPrefixLength (3) characters is accepted and triggers the prefix-match path.
func TestDescribeMatchingResourcesBoundaryPrefix(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

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
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			default:
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			}
		}),
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"}) // "foo" is exactly 3 chars = minPrefixLength

	// The prefix-match path should fire and describe the matching pod.
	if !strings.Contains(buf.String(), d.Output) {
		t.Errorf("expected output for 3-char prefix, got: %q", buf.String())
	}
}

// TestDescribeMatchingResourcesExactlyCap verifies that exactly maxPrefixResourcesForEvents
// (10) matches does NOT trigger the event cap warning.
func TestDescribeMatchingResourcesExactlyCap(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	// Create exactly 10 pods matching prefix "nginx-".
	var podItems []corev1.Pod
	for i := 0; i < 10; i++ {
		podItems = append(podItems, corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("nginx-%03d", i),
				Namespace:       "test",
				ResourceVersion: fmt.Sprintf("%d", 10+i),
			},
			Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
		})
	}
	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "100"},
		Items:    podItems,
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/nginx" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		default:
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.EventList{})}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "nginx"})

	// All 10 pods should be described.
	count := strings.Count(buf.String(), d.Output)
	if count != 10 {
		t.Errorf("expected 10 describe outputs, got %d", count)
	}

	// No warning should be printed — 10 is exactly the cap, not over it.
	if strings.Contains(errbuf.String(), "skipping event queries") {
		t.Errorf("unexpected event cap warning for exactly %d matches: %q", 10, errbuf.String())
	}
}

// TestDescribeMatchingResourcesNoMatch verifies that when no resources match
// the prefix, the original NotFound error is returned.
func TestDescribeMatchingResourcesNoMatch(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

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
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			default:
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			}
		}),
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	var fatalMsg string
	cmdutil.BehaviorOnFatal(func(msg string, code int) {
		fatalMsg = msg
	})
	defer cmdutil.DefaultBehaviorOnFatal()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"}) // "foo" prefix doesn't match "bar-abc"

	if buf.Len() != 0 {
		t.Errorf("expected no output when no resources match prefix, got: %q", buf.String())
	}
	if fatalMsg == "" {
		t.Error("expected fatal error for no matching prefix, got none")
	}
}

// TestDescribeMatchingResourcesBatchWithEvents verifies that when the batch path
// fetches namespace events, matching events appear in the output for each resource.
func TestDescribeMatchingResourcesBatchWithEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

	events := &corev1.EventList{
		Items: []corev1.Event{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "ev-1", Namespace: "test"},
				InvolvedObject: corev1.ObjectReference{
					Name:      "foo-abc",
					Namespace: "test",
					Kind:      "Pod",
				},
				Source:  corev1.EventSource{Component: "kubelet"},
				Message: "Successfully pulled image",
				Type:    corev1.EventTypeNormal,
				Reason:  "Pulled",
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "ev-2", Namespace: "test"},
				InvolvedObject: corev1.ObjectReference{
					Name:      "foo-def",
					Namespace: "test",
					Kind:      "Pod",
				},
				Source:  corev1.EventSource{Component: "kubelet"},
				Message: "Started container",
				Type:    corev1.EventTypeNormal,
				Reason:  "Started",
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/foo" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		case p == "/api/v1/namespaces/test/events" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, events)}, nil
		default:
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.EventList{})}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"})

	output := buf.String()
	// Both pods should be described.
	if count := strings.Count(output, d.Output); count != 2 {
		t.Errorf("expected 2 describe outputs, got %d", count)
	}
	// Events should be appended from the batch fetch.
	if !strings.Contains(output, "Successfully pulled image") {
		t.Errorf("expected foo-abc's event in output, got: %q", output)
	}
	if !strings.Contains(output, "Started container") {
		t.Errorf("expected foo-def's event in output, got: %q", output)
	}
	// ShowEvents should be false (disabled for describer, events appended externally).
	if d.Settings.ShowEvents != false {
		t.Errorf("expected ShowEvents=false in batch path, got %v", d.Settings.ShowEvents)
	}
}

// TestDescribeMatchingResourcesBatchNoEvents verifies that when the batch path
// finds no events for a resource, no "Events: <none>" section is appended.
// This is intentional to avoid adding events to resource types that never show them
// (e.g. SecretDescriber ignores ShowEvents entirely).
func TestDescribeMatchingResourcesBatchNoEvents(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/foo" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		default:
			// Return empty event list — no events in namespace.
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.EventList{})}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"})

	output := buf.String()
	// Both pods described.
	if count := strings.Count(output, d.Output); count != 2 {
		t.Errorf("expected 2 describe outputs, got %d", count)
	}
	// No events section should be appended (no events exist).
	if strings.Contains(output, "Events:") {
		t.Errorf("expected no Events section when batch has no matching events, got: %q", output)
	}
}

// TestDescribeMatchingResourcesBatchFallbackOnEventError verifies that when the
// batch event fetch fails (e.g., RBAC denied), we gracefully fall back to the
// per-resource describe path.
func TestDescribeMatchingResourcesBatchFallbackOnEventError(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "15"},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-abc", Namespace: "test", ResourceVersion: "10"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-def", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	httpClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/namespaces/test/pods/foo" && m == "GET":
			return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
		case p == "/namespaces/test/pods" && m == "GET":
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
		case strings.Contains(p, "events"):
			// Simulate RBAC denied for event listing.
			return &http.Response{StatusCode: http.StatusForbidden, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody(`{"kind":"Status","apiVersion":"v1","status":"Failure","message":"forbidden"}`)}, nil
		default:
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.EventList{})}, nil
		}
	})
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client:               httpClient,
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods", "foo"})

	output := buf.String()
	// Despite event fetch failure, both pods should still be described
	// via the fallback path (per-resource with ShowEvents=true).
	if count := strings.Count(output, d.Output); count != 2 {
		t.Errorf("expected 2 describe outputs after fallback, got %d", count)
	}
	// Fallback path uses ShowEvents=true (the describer handles its own events).
	if d.Settings.ShowEvents != true {
		t.Errorf("expected ShowEvents=true in fallback path, got %v", d.Settings.ShowEvents)
	}
}

// TestDescribeMatchingResourcesEventCapNoWarningWhenEventsDisabled verifies that
// when >10 resources match but ShowEvents is already false, no warning is printed.
func TestDescribeMatchingResourcesEventCapNoWarningWhenEventsDisabled(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	oldFn := describe.DescriberFn
	defer func() {
		describe.DescriberFn = oldFn
	}()
	describe.DescriberFn = d.describerFor

	var podItems []corev1.Pod
	for i := 0; i < 12; i++ {
		podItems = append(podItems, corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("nginx-%03d", i),
				Namespace:       "test",
				ResourceVersion: fmt.Sprintf("%d", 10+i),
			},
			Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyAlways},
		})
	}
	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "100"},
		Items:    podItems,
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods/nginx" && m == "GET":
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			case p == "/namespaces/test/pods" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			default:
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			}
		}),
	}

	streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

	cmd := NewCmdDescribe("kubectl", tf, streams)
	cmd.Flags().Set("show-events", "false")
	cmd.Run(cmd, []string{"pods", "nginx"})

	// All 12 pods described.
	count := strings.Count(buf.String(), d.Output)
	if count != 12 {
		t.Errorf("expected 12 describe outputs, got %d", count)
	}

	// No warning when ShowEvents is already false — nothing to skip.
	if strings.Contains(errbuf.String(), "skipping event queries") {
		t.Errorf("unexpected warning when ShowEvents=false: %q", errbuf.String())
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
