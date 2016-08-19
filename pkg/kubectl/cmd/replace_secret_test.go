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

	"k8s.io/kubernetes/pkg/client/unversioned/fake"
)

func TestReplaceSecretGeneric(t *testing.T) {
	_, secrets := genTestData()
	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "PUT":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdReplaceSecretGeneric(f, buf)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{secrets.Items[0].ObjectMeta.Name})
	expectedOutput := "secret/" + secrets.Items[0].ObjectMeta.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", buf.String(), expectedOutput)
	}
}

func TestReplaceSecretDockerRegistry(t *testing.T) {
	_, secrets := genTestData()
	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "PUT":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdReplaceSecretDockerRegistry(f, buf)
	cmd.Flags().Set("docker-username", "test-user")
	cmd.Flags().Set("docker-password", "test-pass")
	cmd.Flags().Set("docker-email", "test-email")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{secrets.Items[0].ObjectMeta.Name})
	expectedOutput := "secret/" + secrets.Items[0].ObjectMeta.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", buf.String(), expectedOutput)
	}
}

func TestReplaceSecretTLS(t *testing.T) {
	_, secrets := genTestData()
	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			case p == "/namespaces/test/secrets/"+secrets.Items[0].ObjectMeta.Name && m == "PUT":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &secrets.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdReplaceSecretTLS(f, buf)
	cmd.Flags().Set("key", "../../../hack/testdata/tls.key")
	cmd.Flags().Set("cert", "../../../hack/testdata/tls.crt")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{secrets.Items[0].ObjectMeta.Name})
	expectedOutput := "secret/" + secrets.Items[0].ObjectMeta.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", buf.String(), expectedOutput)
	}
}
