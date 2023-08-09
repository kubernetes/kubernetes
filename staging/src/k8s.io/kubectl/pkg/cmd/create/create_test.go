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

package create

import (
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestExtraArgsFail(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	c := NewCmdCreate(f, genericiooptions.NewTestIOStreamsDiscard())
	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
	options := NewCreateOptions(ioStreams)
	if options.Complete(f, c, []string{"rc"}) == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestCreateObject(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, _, rc := cmdtesting.TestData()
	rc.Items[0].Name = "redis-master-controller"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateMultipleObject(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, svc, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../testdata/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateDirectory(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, _, rc := cmdtesting.TestData()
	rc.Items[0].Name = "name"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/replace/legacy")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/name\nreplicationcontroller/name\nreplicationcontroller/name\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
