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

package util

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"syscall"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/utils/exec"
)

func TestMerge(t *testing.T) {
	grace := int64(30)
	tests := []struct {
		obj       runtime.Object
		fragment  string
		expected  runtime.Object
		expectErr bool
	}{
		{
			obj: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s" }`, api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()),
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
		/* TODO: uncomment this test once Merge is updated to use
		strategic-merge-patch. See #8449.
		{
			obj: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
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
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "containers": [ { "name": "c1", "image": "green-image" } ] } }`, api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()),
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
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
			obj: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "volumes": [ {"name": "v1"}, {"name": "v2"} ] } }`, api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()),
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
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
					SchedulerName:                 api.DefaultSchedulerName,
				},
			},
		},
		{
			obj:       &api.Pod{},
			fragment:  "invalid json",
			expected:  &api.Pod{},
			expectErr: true,
		},
		{
			obj:       &api.Service{},
			fragment:  `{ "apiVersion": "badVersion" }`,
			expectErr: true,
		},
		{
			obj: &api.Service{
				Spec: api.ServiceSpec{},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "ports": [ { "port": 0 } ] } }`, api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()),
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
			obj: &api.Service{
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"version": "v1",
					},
				},
			},
			fragment: fmt.Sprintf(`{ "apiVersion": "%s", "spec": { "selector": { "version": "v2" } } }`, api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()),
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
		out, err := Merge(testapi.Default.Codec(), test.obj, test.fragment)
		if !test.expectErr {
			if err != nil {
				t.Errorf("testcase[%d], unexpected error: %v", i, err)
			} else if !apiequality.Semantic.DeepEqual(out, test.expected) {
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

type checkErrTestCase struct {
	err          error
	expectedErr  string
	expectedCode int
}

func TestCheckInvalidErr(t *testing.T) {
	testCheckError(t, []checkErrTestCase{
		{
			errors.NewInvalid(api.Kind("Invalid1"), "invalidation", field.ErrorList{field.Invalid(field.NewPath("field"), "single", "details")}),
			"The Invalid1 \"invalidation\" is invalid: field: Invalid value: \"single\": details\n",
			DefaultErrorExitCode,
		},
		{
			errors.NewInvalid(api.Kind("Invalid2"), "invalidation", field.ErrorList{field.Invalid(field.NewPath("field1"), "multi1", "details"), field.Invalid(field.NewPath("field2"), "multi2", "details")}),
			"The Invalid2 \"invalidation\" is invalid: \n* field1: Invalid value: \"multi1\": details\n* field2: Invalid value: \"multi2\": details\n",
			DefaultErrorExitCode,
		},
		{
			errors.NewInvalid(api.Kind("Invalid3"), "invalidation", field.ErrorList{}),
			"The Invalid3 \"invalidation\" is invalid",
			DefaultErrorExitCode,
		},
		{
			errors.NewInvalid(api.Kind("Invalid4"), "invalidation", field.ErrorList{field.Invalid(field.NewPath("field4"), "multi4", "details"), field.Invalid(field.NewPath("field4"), "multi4", "details")}),
			"The Invalid4 \"invalidation\" is invalid: field4: Invalid value: \"multi4\": details\n",
			DefaultErrorExitCode,
		},
	})
}

func TestCheckNoResourceMatchError(t *testing.T) {
	testCheckError(t, []checkErrTestCase{
		{
			&meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{Resource: "foo"}},
			`the server doesn't have a resource type "foo"`,
			DefaultErrorExitCode,
		},
		{
			&meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{Version: "theversion", Resource: "foo"}},
			`the server doesn't have a resource type "foo" in version "theversion"`,
			DefaultErrorExitCode,
		},
		{
			&meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{Group: "thegroup", Version: "theversion", Resource: "foo"}},
			`the server doesn't have a resource type "foo" in group "thegroup" and version "theversion"`,
			DefaultErrorExitCode,
		},
		{
			&meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{Group: "thegroup", Resource: "foo"}},
			`the server doesn't have a resource type "foo" in group "thegroup"`,
			DefaultErrorExitCode,
		},
	})
}

func TestCheckExitError(t *testing.T) {
	testCheckError(t, []checkErrTestCase{
		{
			exec.CodeExitError{Err: fmt.Errorf("pod foo/bar terminated"), Code: 42},
			"pod foo/bar terminated",
			42,
		},
	})
}

func testCheckError(t *testing.T, tests []checkErrTestCase) {
	var errReturned string
	var codeReturned int
	errHandle := func(err string, code int) {
		errReturned = err
		codeReturned = code
	}

	for _, test := range tests {
		checkErr(test.err, errHandle)

		if errReturned != test.expectedErr {
			t.Fatalf("Got: %s, expected: %s", errReturned, test.expectedErr)
		}
		if codeReturned != test.expectedCode {
			t.Fatalf("Got: %d, expected: %d", codeReturned, test.expectedCode)
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
	defer func() {
		if !t.Failed() {
			os.Remove(tempFile.Name())
		}
	}()
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
