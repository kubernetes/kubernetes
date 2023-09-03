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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
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
