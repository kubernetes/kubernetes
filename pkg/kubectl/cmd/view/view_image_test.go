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

package view

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
)

func TestViewImageLocal(t *testing.T) {
	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion}}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdImage(f, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("local", "true")
	cmd.Flags().Set("no-headers", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := ImageOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../examples/storage/cassandra/cassandra-controller.yaml"}},
		Out:    buf,
		Local:  true,
		Output: "name"}
	err := opts.Run(f, cmd, []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "gcr.io/google-samples/cassandra:v12") {
		t.Errorf("did not view image:\n%s", buf.String())
	}
}

func TestViewMultiImageLocal(t *testing.T) {
	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion}}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdImage(f, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("local", "true")
	cmd.Flags().Set("no-headers", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := ImageOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/view/multi-container.yaml"}},
		Out:    buf,
		Local:  true,
		Output: "name"}
	err := opts.Run(f, cmd, []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "busybox-2\nbusybox-3\nbusybox-1") {
		t.Errorf("did not view image:\n%s", buf.String())
	}
}
