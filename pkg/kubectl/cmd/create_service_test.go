/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateService(t *testing.T) {
	service := &api.Service{}
	service.Name = "my-service"
	f, tf, codec, negSer := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: negSer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: objBody(codec, service)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateServiceClusterIP(f, buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("tcp", "8080:8000")
	cmd.Run(cmd, []string{service.Name})
	expectedOutput := "service/" + service.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateServiceNodePort(t *testing.T) {
	service := &api.Service{}
	service.Name = "my-node-port-service"
	f, tf, codec, negSer := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: negSer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, service)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateServiceNodePort(f, buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("tcp", "30000:8000")
	cmd.Run(cmd, []string{service.Name})
	expectedOutput := "service/" + service.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateServiceExternalName(t *testing.T) {
	service := &api.Service{}
	service.Name = "my-external-name-service"
	f, tf, codec, negSer := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: negSer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, service)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateServiceExternalName(f, buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("external-name", "name")
	cmd.Run(cmd, []string{service.Name})
	expectedOutput := "service/" + service.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}
