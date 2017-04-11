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

package cmd

import (
	"bytes"
	"fmt"
	"net/http"
	"testing"

	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestDescribeUnknownSchemaObject(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	f, tf, codec, _ := cmdtesting.NewTestFactory()
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, cmdtesting.NewInternalType("", "", "foo"))},
	}
	tf.Namespace = "non-default"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Run(cmd, []string{"type", "foo"})

	if d.Name != "foo" || d.Namespace != "" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != fmt.Sprintf("%s", d.Output) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestDescribeUnknownNamespacedSchemaObject(t *testing.T) {
	d := &testDescriber{Output: "test output"}
	f, tf, codec, _ := cmdtesting.NewTestFactory()
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, cmdtesting.NewInternalNamespacedType("", "", "foo", "non-default"))},
	}
	tf.Namespace = "non-default"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Run(cmd, []string{"namespacedtype", "foo"})

	if d.Name != "foo" || d.Namespace != "non-default" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != fmt.Sprintf("%s", d.Output) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeObject(t *testing.T) {
	_, _, rc := testData()
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	d := &testDescriber{Output: "test output"}
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Run(cmd, []string{})

	if d.Name != "redis-master" || d.Namespace != "test" {
		t.Errorf("unexpected describer: %#v", d)
	}

	if buf.String() != fmt.Sprintf("%s", d.Output) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeListObjects(t *testing.T) {
	pods, _, _ := testData()
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	d := &testDescriber{Output: "test output"}
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}

	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Run(cmd, []string{"pods"})
	if buf.String() != fmt.Sprintf("%s\n\n%s", d.Output, d.Output) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDescribeObjectShowEvents(t *testing.T) {
	pods, _, _ := testData()
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	d := &testDescriber{Output: "test output"}
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}

	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Flags().Set("show-events", "true")
	cmd.Run(cmd, []string{"pods"})
	if d.Settings.ShowEvents != true {
		t.Errorf("ShowEvents = true expected, got ShowEvents = %v", d.Settings.ShowEvents)
	}
}

func TestDescribeObjectSkipEvents(t *testing.T) {
	pods, _, _ := testData()
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	d := &testDescriber{Output: "test output"}
	tf.Describer = d
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}

	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	buferr := bytes.NewBuffer([]byte{})
	cmd := NewCmdDescribe(f, buf, buferr)
	cmd.Flags().Set("show-events", "false")
	cmd.Run(cmd, []string{"pods"})
	if d.Settings.ShowEvents != false {
		t.Errorf("ShowEvents = false expected, got ShowEvents = %v", d.Settings.ShowEvents)
	}
}
