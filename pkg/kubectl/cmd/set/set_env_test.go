/*
Copyright 2017 The Kubernetes Authors.

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

package set

import (
	"bytes"
	"net/http"
	"os"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
)

func TestSetEnvLocal(t *testing.T) {
	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion}}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdEnv(f, os.Stdin, buf, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("local", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := EnvOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../examples/storage/cassandra/cassandra-controller.yaml"}},
		Out:   buf,
		Local: true}
	err := opts.Complete(f, cmd, []string{"env=prod"})
	if err == nil {
		err = opts.RunEnv(f)
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "replicationcontrollers/cassandra") {
		t.Errorf("did not set env: %s", buf.String())
	}
}

func TestSetMultiResourcesEnvLocal(t *testing.T) {
	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion}}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdEnv(f, os.Stdin, buf, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("local", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := EnvOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"}},
		Out:   buf,
		Local: true}
	err := opts.Complete(f, cmd, []string{"env=prod"})
	if err == nil {
		err = opts.RunEnv(f)
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedOut := "replicationcontrollers/first-rc\nreplicationcontrollers/second-rc\n"
	if buf.String() != expectedOut {
		t.Errorf("expected out:\n%s\nbut got:\n%s", expectedOut, buf.String())
	}
}
