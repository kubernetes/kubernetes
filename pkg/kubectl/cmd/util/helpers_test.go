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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"syscall"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

func TestMerge(t *testing.T) {
	grace := int64(30)
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
			fragment: fmt.Sprintf(`{ "apiVersion": "%s" }`, testapi.Default.Version()),
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
		/* TODO: uncomment this test once Merge is updated to use
		strategic-merge-patch. See #8449.
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						api.Container{
							Name:  "c1",
							Image: "red-image",
						},
						api.Container{
							Name:  "c2",
							Image: "blue-image",
						},
					},
				},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "containers": [ { "name": "c1", "image": "green-image" } ] } }`, testapi.Default.Version()),
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						api.Container{
							Name:  "c1",
							Image: "green-image",
						},
						api.Container{
							Name:  "c2",
							Image: "blue-image",
						},
					},
				},
			},
		}, */
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "volumes": [ {"name": "v1"}, {"name": "v2"} ] } }`, testapi.Default.Version()),
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
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &api.PodSecurityContext{},
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
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "ports": [ { "port": 0 } ] } }`, testapi.Default.Version()),
			expected: &api.Service{
				Spec: api.ServiceSpec{
					SessionAffinity: "None",
					Type:            api.ServiceTypeClusterIP,
					Ports: []api.ServicePort{
						{
							Protocol: api.ProtocolTCP,
							Port:     0,
						},
					},
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
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "selector": { "version": "v2" } } }`, testapi.Default.Version()),
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

func TestCheckInvalidErr(t *testing.T) {
	tests := []struct {
		err      error
		expected string
	}{
		{
			errors.NewInvalid("Invalid1", "invalidation", fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("Cause", "single", "details")}),
			`Error from server: Invalid1 "invalidation" is invalid: Cause: invalid value 'single', Details: details`,
		},
		{
			errors.NewInvalid("Invalid2", "invalidation", fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("Cause", "multi1", "details"), fielderrors.NewFieldInvalid("Cause", "multi2", "details")}),
			`Error from server: Invalid2 "invalidation" is invalid: [Cause: invalid value 'multi1', Details: details, Cause: invalid value 'multi2', Details: details]`,
		},
		{
			errors.NewInvalid("Invalid3", "invalidation", fielderrors.ValidationErrorList{}),
			`Error from server: Invalid3 "invalidation" is invalid: <nil>`,
		},
	}

	var errReturned string
	errHandle := func(err string) {
		errReturned = err
	}

	for _, test := range tests {
		checkErr(test.err, errHandle)

		if errReturned != test.expected {
			t.Fatalf("Got: %s, expected: %s", errReturned, test.expected)
		}
	}
}

func TestDumpReaderToFile(t *testing.T) {
	testString := "TEST STRING"
	tempFile, err := ioutil.TempFile("", "hlpers_test_dump_")
	if err != nil {
		t.Errorf("unexpected error setting up a temporary file %v", err)
	}
	defer syscall.Unlink(tempFile.Name())
	defer tempFile.Close()
	err = DumpReaderToFile(strings.NewReader(testString), tempFile.Name())
	if err != nil {
		t.Errorf("error in DumpReaderToFile: %v", err)
	}
	data, err := ioutil.ReadFile(tempFile.Name())
	if err != nil {
		t.Errorf("error when reading %s: %v", tempFile.Name(), err)
	}
	stringData := string(data)
	if stringData != testString {
		t.Fatalf("Wrong file content %s != %s", testString, stringData)
	}
}
