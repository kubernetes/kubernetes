/*
Copyright 2015 The Kubernetes Authors.

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
	"net/http"
	"strings"
	"testing"

	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/printers"
)

func TestPatchObject(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				obj := svc.Items[0]

				// ensure patched object reflects successful
				// patch edits from the client
				if m == "PATCH" {
					obj.Spec.Type = "NodePort"
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &obj)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"services/frontend"})

	// uses the name from the response
	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchObjectFromFile(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Run(cmd, []string{})

	// uses the name from the response
	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchNoop(t *testing.T) {
	_, svc, _ := testData()
	getObject := &svc.Items[0]
	patchObject := &svc.Items[0]

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && m == "PATCH":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, patchObject)}, nil
			case p == "/namespaces/test/services/frontend" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, getObject)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"

	// Patched
	{
		patchObject = patchObject.DeepCopy()
		if patchObject.Annotations == nil {
			patchObject.Annotations = map[string]string{}
		}
		patchObject.Annotations["foo"] = "bar"
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdPatch(f, buf)
		cmd.Flags().Set("namespace", "test")
		cmd.Flags().Set("patch", `{"metadata":{"annotations":{"foo":"bar"}}}`)
		cmd.Run(cmd, []string{"services", "frontend"})
		if buf.String() != "service \"baz\" patched\n" {
			t.Errorf("unexpected output: %s", buf.String())
		}
	}
}

func TestPatchObjectFromFileOutput(t *testing.T) {
	_, svc, _ := testData()

	svcCopy := svc.Items[0].DeepCopy()
	if svcCopy.Labels == nil {
		svcCopy.Labels = map[string]string{}
	}
	svcCopy.Labels["post-patch"] = "post-patch-value"

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &printers.YAMLPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == "PATCH":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svcCopy)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Run(cmd, []string{})

	t.Log(buf.String())
	// make sure the value returned by the server is used
	if !strings.Contains(buf.String(), "post-patch: post-patch-value") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
