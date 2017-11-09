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

package set

import (
	"bytes"
	"net/http"
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

func TestImageLocal(t *testing.T) {
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
	cmd := NewCmdImage(f, buf, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("local", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := ImageOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../examples/storage/cassandra/cassandra-controller.yaml"}},
		Out:   buf,
		Local: true}
	err := opts.Complete(f, cmd, []string{"cassandra=thingy"})
	if err == nil {
		err = opts.Validate()
	}
	if err == nil {
		err = opts.Run()
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "replicationcontrollers/cassandra") {
		t.Errorf("did not set image: %s", buf.String())
	}
}

func TestSetImageValidation(t *testing.T) {
	testCases := []struct {
		name         string
		imageOptions *ImageOptions
		expectErr    string
	}{
		{
			name:         "test resource < 1 and filenames empty",
			imageOptions: &ImageOptions{},
			expectErr:    "[one or more resources must be specified as <resource> <name> or <resource>/<name>, at least one image update is required]",
		},
		{
			name: "test containerImages < 1",
			imageOptions: &ImageOptions{
				Resources: []string{"a", "b", "c"},

				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
			},
			expectErr: "at least one image update is required",
		},
		{
			name: "test containerImages > 1 and all containers are already specified by *",
			imageOptions: &ImageOptions{
				Resources: []string{"a", "b", "c"},
				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
				ContainerImages: map[string]string{
					"test": "test",
					"*":    "test",
				},
			},
			expectErr: "all containers are already specified by *, but saw more than one container_name=container_image pairs",
		},
		{
			name: "sucess case",
			imageOptions: &ImageOptions{
				Resources: []string{"a", "b", "c"},
				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
				ContainerImages: map[string]string{
					"test": "test",
				},
			},
			expectErr: "",
		},
	}
	for _, testCase := range testCases {
		err := testCase.imageOptions.Validate()
		if err != nil {
			if err.Error() != testCase.expectErr {
				t.Errorf("[%s]:expect err:%s got err:%s", testCase.name, testCase.expectErr, err.Error())
			}
		}
		if err == nil && (testCase.expectErr != "") {
			t.Errorf("[%s]:expect err:%s got err:%v", testCase.name, testCase.expectErr, err)
		}
	}
}

func TestSetMultiResourcesImageLocal(t *testing.T) {
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
	cmd := NewCmdImage(f, buf, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("local", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}

	opts := ImageOptions{FilenameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"}},
		Out:   buf,
		Local: true}
	err := opts.Complete(f, cmd, []string{"*=thingy"})
	if err == nil {
		err = opts.Validate()
	}
	if err == nil {
		err = opts.Run()
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedOut := "replicationcontrollers/first-rc\nreplicationcontrollers/second-rc\n"
	if buf.String() != expectedOut {
		t.Errorf("expected out:\n%s\nbut got:\n%s", expectedOut, buf.String())
	}
}
