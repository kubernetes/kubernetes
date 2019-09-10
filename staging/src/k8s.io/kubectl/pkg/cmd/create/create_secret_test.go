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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCreateSecretGeneric(t *testing.T) {
	secretObject := &v1.Secret{
		Data: map[string][]byte{
			"password": []byte("includes,comma"),
			"username": []byte("test_user"),
		},
	}
	secretObject.Name = "my-secret"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/secrets" && m == "POST":
				return &http.Response{StatusCode: 201, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, secretObject)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreateSecretGeneric(tf, ioStreams)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("from-literal", "password=includes,comma")
	cmd.Flags().Set("from-literal", "username=test_user")
	cmd.Run(cmd, []string{secretObject.Name})
	expectedOutput := "secret/" + secretObject.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateSecretDockerRegistry(t *testing.T) {
	secretObject := &v1.Secret{}
	secretObject.Name = "my-secret"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/secrets" && m == "POST":
				return &http.Response{StatusCode: 201, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, secretObject)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreateSecretDockerRegistry(tf, ioStreams)
	cmd.Flags().Set("docker-username", "test-user")
	cmd.Flags().Set("docker-password", "test-pass")
	cmd.Flags().Set("docker-email", "test-email")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{secretObject.Name})
	expectedOutput := "secret/" + secretObject.Name + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", buf.String(), expectedOutput)
	}
}
