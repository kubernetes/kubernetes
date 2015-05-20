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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"syscall"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestMerge(t *testing.T) {
	tests := []struct {
		obj       runtime.Object
		fragment  string
		expected  runtime.Object
		expectErr bool
		kind      string
	}{
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1" }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "id": "baz", "desiredState": { "host": "bar" } }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "baz",
				},
				Spec: api.PodSpec{
					Host:          "bar",
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta3", "spec": { "volumes": [ {"name": "v1"}, {"name": "v2"} ] } }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{
							Name:         "v1",
							VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
						},
						{
							Name:         "v2",
							VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
		{
			kind:      "Pod",
			obj:       &api.Pod{},
			fragment:  "invalid json",
			expected:  &api.Pod{},
			expectErr: true,
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "id": null}`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
		{
			kind:      "Service",
			obj:       &api.Service{},
			fragment:  `{ "apiVersion": "badVersion" }`,
			expectErr: true,
		},
		{
			kind: "Service",
			obj: &api.Service{
				Spec: api.ServiceSpec{},
			},
			fragment: `{ "apiVersion": "v1beta1", "port": 0 }`,
			expected: &api.Service{
				Spec: api.ServiceSpec{
					SessionAffinity: "None",
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			kind: "Service",
			obj: &api.Service{
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"version": "v1",
					},
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "selector": { "version": "v2" } }`,
			expected: &api.Service{
				Spec: api.ServiceSpec{
					SessionAffinity: "None",
					Type:            api.ServiceTypeClusterIP,
					Selector: map[string]string{
						"version": "v2",
					},
				},
			},
		},
	}

	for i, test := range tests {
		out, err := Merge(test.obj, test.fragment, test.kind)
		if !test.expectErr {
			if err != nil {
				t.Errorf("testcase[%d], unexpected error: %v", i, err)
			} else if !reflect.DeepEqual(out, test.expected) {
				t.Errorf("\n\ntestcase[%d]\nexpected:\n%+v\nsaw:\n%+v", i, test.expected, out)
			}
		}
		if test.expectErr && err == nil {
			t.Errorf("testcase[%d], unexpected non-error", i)
		}
	}
}

type fileHandler struct {
	data []byte
}

func (f *fileHandler) ServeHTTP(res http.ResponseWriter, req *http.Request) {
	if req.URL.Path == "/error" {
		res.WriteHeader(http.StatusNotFound)
		return
	}
	res.WriteHeader(http.StatusOK)
	res.Write(f.data)
}

func TestReadConfigData(t *testing.T) {
	httpData := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	server := httptest.NewServer(&fileHandler{data: httpData})

	fileData := []byte{11, 12, 13, 14, 15, 16, 17, 18, 19}
	f, err := ioutil.TempFile("", "config")
	if err != nil {
		t.Errorf("unexpected error setting up config file")
		t.Fail()
	}
	defer syscall.Unlink(f.Name())
	ioutil.WriteFile(f.Name(), fileData, 0644)
	// TODO: test TLS here, requires making it possible to inject the HTTP client.

	tests := []struct {
		config    string
		data      []byte
		expectErr bool
	}{
		{
			config: server.URL,
			data:   httpData,
		},
		{
			config:    server.URL + "/error",
			expectErr: true,
		},
		{
			config:    "http://some.non.existent.foobar",
			expectErr: true,
		},
		{
			config: f.Name(),
			data:   fileData,
		},
		{
			config:    "some-non-existent-file",
			expectErr: true,
		},
		{
			config:    "",
			expectErr: true,
		},
	}
	for _, test := range tests {
		dataOut, err := ReadConfigData(test.config)
		if err != nil && !test.expectErr {
			t.Errorf("unexpected err: %v for %s", err, test.config)
		}
		if err == nil && test.expectErr {
			t.Errorf("unexpected non-error for %s", test.config)
		}
		if !test.expectErr && !reflect.DeepEqual(test.data, dataOut) {
			t.Errorf("unexpected data: %v, expected %v", dataOut, test.data)
		}
	}
}
