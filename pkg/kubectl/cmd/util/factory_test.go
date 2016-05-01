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
	"os/user"
	"path"
	"sort"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flag"
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
				TypeMeta:   unversioned.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "svc=test",
			err:      nil,
		},
		{
			name: "empty labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", Labels: map[string]string{}},
				TypeMeta:   unversioned.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
		{
			name: "nil labels",
			object: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "zen", Namespace: "test", Labels: nil},
				TypeMeta:   unversioned.TypeMeta{Kind: "Service", APIVersion: "v1"},
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

func TestCanBeExposed(t *testing.T) {
	factory := NewFactory(nil)
	tests := []struct {
		kind      unversioned.GroupKind
		expectErr bool
	}{
		{
			kind:      api.Kind("ReplicationController"),
			expectErr: false,
		},
		{
			kind:      api.Kind("Node"),
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := factory.CanBeExposed(test.kind)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestFlagUnderscoreRenaming(t *testing.T) {
	factory := NewFactory(nil)

	factory.flags.SetNormalizeFunc(flag.WordSepNormalizeFunc)
	factory.flags.Bool("valid_flag", false, "bool value")

	// In case of failure of this test check this PR: spf13/pflag#23
	if factory.flags.Lookup("valid_flag").Name != "valid-flag" {
		t.Fatalf("Expected flag name to be valid-flag, got %s", factory.flags.Lookup("valid_flag").Name)
	}
}

func loadSchemaForTest() (validation.Schema, error) {
	pathToSwaggerSpec := "../../../../api/swagger-spec/" + testapi.Default.GroupVersion().Version + ".json"
	data, err := ioutil.ReadFile(pathToSwaggerSpec)
	if err != nil {
		return nil, err
	}
	return validation.NewSwaggerSchemaFromBytes(data)
}

func TestRefetchSchemaWhenValidationFails(t *testing.T) {
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

	c := &fake.RESTClient{
		Codec: testapi.Default.Codec(),
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
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

	fullDir, err := substituteUserHome(dir)
	if err != nil {
		t.Errorf("Error getting fullDir: %v", err)
		t.FailNow()
	}
	cacheFile := path.Join(fullDir, "foo", "bar", schemaFileName)
	err = writeSchemaFile(output, fullDir, cacheFile, "foo", "bar")
	if err != nil {
		t.Errorf("Error building old cache schema: %v", err)
		t.FailNow()
	}

	obj := &extensions.Deployment{}
	data, err := runtime.Encode(testapi.Extensions.Codec(), obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}

	// Re-get request, should use HTTP and write
	if getSchemaAndValidate(c, data, "foo", "bar", dir); err != nil {
		t.Errorf("unexpected error validating: %v", err)
	}
	if requests["/swaggerapi/foo/bar"] != 1 {
		t.Errorf("expected 1 schema request, saw: %d", requests["/swaggerapi/foo/bar"])
	}
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

	c := &fake.RESTClient{
		Codec: testapi.Default.Codec(),
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
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
	data, err := runtime.Encode(testapi.Default.Codec(), obj)
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
	if requests["/swaggerapi/foo/bar"] != 2 {
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

func TestSubstitueUser(t *testing.T) {
	usr, err := user.Current()
	if err != nil {
		t.Logf("SKIPPING TEST: unexpected error: %v", err)
		return
	}
	tests := []struct {
		input     string
		expected  string
		expectErr bool
	}{
		{input: "~/foo", expected: path.Join(os.Getenv("HOME"), "foo")},
		{input: "~" + usr.Username + "/bar", expected: usr.HomeDir + "/bar"},
		{input: "/foo/bar", expected: "/foo/bar"},
		{input: "~doesntexit/bar", expectErr: true},
	}
	for _, test := range tests {
		output, err := substituteUserHome(test.input)
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if output != test.expected {
			t.Errorf("expected: %s, saw: %s", test.expected, output)
		}
	}
}
