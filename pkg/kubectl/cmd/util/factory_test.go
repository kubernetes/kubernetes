/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/validation"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func TestNewFactoryDefaultFlagBindings(t *testing.T) {
	factory := NewFactory(nil)

	if !factory.flags.HasFlags() {
		t.Errorf("Expected flags, but didn't get any")
	}
}

func TestNewFactoryNoFlagBindings(t *testing.T) {
	clientConfig := clientcmd.NewDefaultClientConfig(*clientcmdapi.NewConfig(), &clientcmd.ConfigOverrides{})
	factory := NewFactory(clientConfig)

	if factory.flags.HasFlags() {
		t.Errorf("Expected zero flags, but got %v", factory.flags)
	}
}

func TestPodSelectorForObject(t *testing.T) {
	f := NewFactory(nil)

	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	expected := "foo=bar"
	got, err := f.PodSelectorForObject(svc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected != got {
		t.Fatalf("Selector mismatch! Expected %s, got %s", expected, got)
	}
}

func TestPortsForObject(t *testing.T) {
	f := NewFactory(nil)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
						},
					},
				},
			},
		},
	}

	expected := []string{"101"}
	got, err := f.PortsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(expected) != len(got) {
		t.Fatalf("Ports size mismatch! Expected %d, got %d", len(expected), len(got))
	}

	sort.Strings(expected)
	sort.Strings(got)

	for i, port := range got {
		if port != expected[i] {
			t.Fatalf("Port mismatch! Expected %s, got %s", expected[i], port)
		}
	}
}

func TestLabelsForObject(t *testing.T) {
	f := NewFactory(nil)

	tests := []struct {
		name     string
		object   runtime.Object
		expected string
		err      error
	}{
		{
			name: "successful re-use of labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "svc=test",
			err:      nil,
		},
		{
			name: "empty labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", Labels: map[string]string{}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
		{
			name: "nil labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "zen", Namespace: "test", Labels: nil},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
	}

	for _, test := range tests {
		gotLabels, err := f.LabelsForObject(test.object)
		if err != test.err {
			t.Fatalf("%s: Error mismatch: Expected %v, got %v", test.name, test.err, err)
		}
		got := kubectl.MakeLabels(gotLabels)
		if test.expected != got {
			t.Fatalf("%s: Labels mismatch! Expected %s, got %s", test.name, test.expected, got)
		}

	}
}

func TestFlagUnderscoreRenaming(t *testing.T) {
	factory := NewFactory(nil)

	factory.flags.SetNormalizeFunc(util.WordSepNormalizeFunc)
	factory.flags.Bool("valid_flag", false, "bool value")

	// In case of failure of this test check this PR: spf13/pflag#23
	if factory.flags.Lookup("valid_flag").Name != "valid-flag" {
		t.Fatalf("Expected flag name to be valid-flag, got %s", factory.flags.Lookup("valid_flag").Name)
	}
}

func loadSchemaForTest() (validation.Schema, error) {
	pathToSwaggerSpec := "../../../../api/swagger-spec/" + testapi.Default.Version() + ".json"
	data, err := ioutil.ReadFile(pathToSwaggerSpec)
	if err != nil {
		return nil, err
	}
	return validation.NewSwaggerSchemaFromBytes(data)
}

func TestValidateCachesSchema(t *testing.T) {
	schema, err := loadSchemaForTest()
	if err != nil {
		t.Errorf("Error loading schema: %v", err)
		t.FailNow()
	}
	output, err := json.Marshal(schema)
	if err != nil {
		t.Errorf("Error serializing schema: %v", err)
		t.FailNow()
	}
	requests := map[string]int{}

	c := &client.FakeRESTClient{
		Codec: testapi.Default.Codec(),
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/swaggerapi") && m == "GET":
				requests[p] = requests[p] + 1
				return &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewBuffer(output))}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	dir := os.TempDir() + "/schemaCache"
	os.RemoveAll(dir)

	obj := &api.Pod{}
	data, err := testapi.Default.Codec().Encode(obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}

	// Initial request, should use HTTP and write
	if getSchemaAndValidate(c, data, "foo", "bar", dir); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if _, err := os.Stat(path.Join(dir, "foo", "bar", schemaFileName)); err != nil {
		t.Errorf("unexpected missing cache file: %v", err)
	}
	if requests["/swaggerapi/foo/bar"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo/bar"])
	}

	// Same version and group, should skip HTTP
	if getSchemaAndValidate(c, data, "foo", "bar", dir); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if requests["/swaggerapi/foo/bar"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo/bar"])
	}

	// Different API group, should go to HTTP and write
	if getSchemaAndValidate(c, data, "foo", "baz", dir); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if _, err := os.Stat(path.Join(dir, "foo", "baz", schemaFileName)); err != nil {
		t.Errorf("unexpected missing cache file: %v", err)
	}
	if requests["/swaggerapi/foo/baz"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo/baz"])
	}

	// Different version, should go to HTTP and write
	if getSchemaAndValidate(c, data, "foo2", "bar", dir); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if _, err := os.Stat(path.Join(dir, "foo2", "bar", schemaFileName)); err != nil {
		t.Errorf("unexpected missing cache file: %v", err)
	}
	if requests["/swaggerapi/foo2/bar"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo2/bar"])
	}

	// No cache dir, should go straight to HTTP and not write
	if getSchemaAndValidate(c, data, "foo", "blah", ""); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if requests["/swaggerapi/foo/blah"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo/blah"])
	}
	if _, err := os.Stat(path.Join(dir, "foo", "blah", schemaFileName)); err == nil || !os.IsNotExist(err) {
		t.Errorf("unexpected cache file error: %v", err)
	}
}
