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

package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	scheme "k8s.io/kubernetes/pkg/api/legacyscheme"
)

type testcase struct {
	name          string
	file          string
	outputVersion string
	fields        []checkField
}

type checkField struct {
	expected string
}

func TestConvertObject(t *testing.T) {
	testcases := []testcase{
		{
			name:          "apps deployment to extensions deployment",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/appsdeployment.yaml",
			outputVersion: "extensions/v1beta1",
			fields: []checkField{
				{
					expected: "apiVersion: extensions/v1beta1",
				},
			},
		},
		{
			name:          "extensions deployment to apps deployment",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/extensionsdeployment.yaml",
			outputVersion: "apps/v1beta2",
			fields: []checkField{
				{
					expected: "apiVersion: apps/v1beta2",
				},
			},
		},
		{
			name:          "v1beta1 Ingress to extensions Ingress",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/v1beta1ingress.yaml",
			outputVersion: "extensions/v1beta1",
			fields: []checkField{
				{
					expected: "apiVersion: extensions/v1beta1",
				},
			},
		},
		{
			name:          "converting multiple including service to neworking.k8s.io/v1",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/serviceandingress.yaml",
			outputVersion: "networking.k8s.io/v1",
			fields: []checkField{
				{
					expected: "apiVersion: networking.k8s.io/v1",
				},
			},
		},
	}

	for _, tc := range testcases {
		for _, field := range tc.fields {
			t.Run(fmt.Sprintf("%s %s", tc.name, field), func(t *testing.T) {
				tf := cmdtesting.NewTestFactory().WithNamespace("test")
				defer tf.Cleanup()

				tf.UnstructuredClient = &fake.RESTClient{
					Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}),
				}

				buf := bytes.NewBuffer([]byte{})
				cmd := NewCmdConvert(tf, genericiooptions.IOStreams{Out: buf, ErrOut: buf})
				cmd.Flags().Set("filename", tc.file)
				cmd.Flags().Set("output-version", tc.outputVersion)
				cmd.Flags().Set("local", "true")
				cmd.Flags().Set("output", "yaml")
				cmd.Run(cmd, []string{})
				if !strings.Contains(buf.String(), field.expected) {
					t.Errorf("unexpected output when converting %s to %q, expected: %q, but got %q", tc.file, tc.outputVersion, field.expected, buf.String())
				}
			})
		}
	}
}

func TestAsVersionedObjectsSetsUnknownMetadata(t *testing.T) {
	unregistered := &fakeUnregisteredObject{
		TypeMeta: runtime.TypeMeta{
			APIVersion: "example.com/v1",
			Kind:       "Example",
		},
		Data: "value",
	}
	if _, _, err := scheme.Scheme.ObjectKinds(unregistered); !runtime.IsNotRegisteredError(err) {
		t.Fatalf("expected not registered error, got %v", err)
	}
	info := &resource.Info{Object: unregistered}
	encoder := &stubJSONEncoder{identifier: runtime.Identifier("test-encoder")}
	streams := genericiooptions.NewTestIOStreamsDiscard()
	objs, err := asVersionedObjects([]*resource.Info{info}, schema.GroupVersion{Group: "example.com", Version: "v1"}, encoder, streams)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 1 {
		t.Fatalf("expected 1 object, got %d", len(objs))
	}
	unknown, ok := objs[0].(*runtime.Unknown)
	if !ok {
		t.Fatalf("expected runtime.Unknown, got %T", objs[0])
	}
	if unknown.ContentType != runtime.ContentTypeJSON {
		t.Errorf("expected ContentType %q, got %q", runtime.ContentTypeJSON, unknown.ContentType)
	}
	if unknown.ContentEncoding != "" {
		t.Errorf("expected ContentEncoding %q, got %q", "", unknown.ContentEncoding)
	}
	var decoded map[string]any
	if err := json.Unmarshal(unknown.Raw, &decoded); err != nil {
		t.Fatalf("unable to decode raw payload: %v", err)
	}
	if decoded["kind"] != "Example" {
		t.Errorf("expected kind %q, got %v", "Example", decoded["kind"])
	}
}

type fakeUnregisteredObject struct {
	runtime.TypeMeta
	Data string `json:"data"`
}

func (f *fakeUnregisteredObject) DeepCopyObject() runtime.Object {
	copy := *f
	return &copy
}

type stubJSONEncoder struct {
	identifier runtime.Identifier
}

func (s *stubJSONEncoder) Encode(obj runtime.Object, w io.Writer) error {
	return json.NewEncoder(w).Encode(obj)
}

func (s *stubJSONEncoder) Identifier() runtime.Identifier {
	return s.identifier
}
