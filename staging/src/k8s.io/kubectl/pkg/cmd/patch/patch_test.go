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

package patch

import (
	"net/http"
	"strings"
	"testing"

	jsonpath "github.com/exponent-io/jsonpath"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestPatchObject(t *testing.T) {
	_, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				obj := svc.Items[0]

				// ensure patched object reflects successful
				// patch edits from the client
				if m == "PATCH" {
					obj.Spec.Type = "NodePort"
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &obj)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	stream, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdPatch(tf, stream)
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
	_, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	stream, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdPatch(tf, stream)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("filename", "../../../testdata/frontend-service.yaml")
	cmd.Run(cmd, []string{})

	// uses the name from the response
	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchNoop(t *testing.T) {
	_, svc, _ := cmdtesting.TestData()
	getObject := &svc.Items[0]
	patchObject := &svc.Items[0]

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && m == "PATCH":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, patchObject)}, nil
			case p == "/namespaces/test/services/frontend" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, getObject)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	// Patched
	{
		patchObject = patchObject.DeepCopy()
		if patchObject.Annotations == nil {
			patchObject.Annotations = map[string]string{}
		}
		patchObject.Annotations["foo"] = "bar"
		stream, _, buf, _ := genericiooptions.NewTestIOStreams()
		cmd := NewCmdPatch(tf, stream)
		cmd.Flags().Set("namespace", "test")
		cmd.Flags().Set("patch", `{"metadata":{"annotations":{"foo":"bar"}}}`)
		cmd.Run(cmd, []string{"services", "frontend"})
		if buf.String() != "service/baz patched\n" {
			t.Errorf("unexpected output: %s", buf.String())
		}
	}
}

func TestPatchObjectFromFileOutput(t *testing.T) {
	_, svc, _ := cmdtesting.TestData()

	svcCopy := svc.Items[0].DeepCopy()
	if svcCopy.Labels == nil {
		svcCopy.Labels = map[string]string{}
	}
	svcCopy.Labels["post-patch"] = "post-patch-value"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == "PATCH":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svcCopy)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	stream, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdPatch(tf, stream)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "yaml")
	cmd.Flags().Set("filename", "../../../testdata/frontend-service.yaml")
	cmd.Run(cmd, []string{})

	t.Log(buf.String())
	// make sure the value returned by the server is used
	if !strings.Contains(buf.String(), "post-patch: post-patch-value") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchSubresource(t *testing.T) {
	pod := cmdtesting.SubresourceTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	expectedStatus := corev1.PodRunning

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods/foo/status" && (m == "PATCH" || m == "GET"):
				obj := pod

				// ensure patched object reflects successful
				// patch edits from the client
				if m == "PATCH" {
					obj.Status.Phase = expectedStatus
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, obj)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	stream, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdPatch(tf, stream)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"status":{"phase":"Running"}}`)
	cmd.Flags().Set("output", "json")
	cmd.Flags().Set("subresource", "status")
	cmd.Run(cmd, []string{"pod/foo"})

	decoder := jsonpath.NewDecoder(buf)
	var actualStatus corev1.PodPhase
	decoder.SeekTo("status", "phase")
	decoder.Decode(&actualStatus)
	// check the status.phase value is updated in the response
	if actualStatus != expectedStatus {
		t.Errorf("unexpected pod status to be set to %s got: %s", expectedStatus, actualStatus)
	}
}
