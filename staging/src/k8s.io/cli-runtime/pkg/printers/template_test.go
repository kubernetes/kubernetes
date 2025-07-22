/*
Copyright 2018 The Kubernetes Authors.

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

package printers

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestTemplate(t *testing.T) {
	testCase := []struct {
		name      string
		template  string
		obj       runtime.Object
		expectOut string
		expectErr func(error) (string, bool)
	}{
		{
			name:     "support base64 decoding of secret data",
			template: "{{ .data.username | base64decode }}",
			obj: &v1.Secret{
				Data: map[string][]byte{
					"username": []byte("hunter"),
				},
			},
			expectOut: "hunter",
		},
		{
			name:     "test error path for base64 decoding",
			template: "{{ .data.username | base64decode }}",
			obj:      &badlyMarshaledSecret{},
			expectErr: func(err error) (string, bool) {
				matched := strings.Contains(err.Error(), "base64 decode")
				return "a base64 decode error", matched
			},
		},
		{
			name:     "template 'eq' should not throw error for numbers",
			template: "{{ eq .count 1}}",
			obj: &v1.Event{
				Count: 1,
			},
			expectOut: "true",
		},
	}
	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			buffer := &bytes.Buffer{}

			p, err := NewGoTemplatePrinter([]byte(test.template))
			if err != nil {
				if test.expectErr == nil {
					t.Errorf("[%s]expected success but got:\n %v\n", test.name, err)
					return
				}
				if expected, ok := test.expectErr(err); !ok {
					t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", test.name, expected, err)
				}
				return
			}

			err = p.PrintObj(test.obj, buffer)
			if err != nil {
				if test.expectErr == nil {
					t.Errorf("[%s]expected success but got:\n %v\n", test.name, err)
					return
				}
				if expected, ok := test.expectErr(err); !ok {
					t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", test.name, expected, err)
				}
				return
			}

			if test.expectErr != nil {
				t.Errorf("[%s]expect:\n error\n but got:\n no error\n", test.name)
				return
			}

			if test.expectOut != buffer.String() {
				t.Errorf("[%s]expect:\n %v\n but got:\n %v\n", test.name, test.expectOut, buffer.String())
			}
		})
	}
}

type badlyMarshaledSecret struct {
	v1.Secret
}

func (a badlyMarshaledSecret) MarshalJSON() ([]byte, error) {
	return []byte(`{"apiVersion":"v1","data":{"username":"--THIS IS NOT BASE64--"},"kind":"Secret"}`), nil
}

func TestTemplateStrings(t *testing.T) {
	// This unit tests the "exists" function as well as the template from update.sh
	table := map[string]struct {
		pod    v1.Pod
		expect string
	}{
		"nilInfo":   {v1.Pod{}, "false"},
		"emptyInfo": {v1.Pod{Status: v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{}}}, "false"},
		"fooExists": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
						},
					},
				},
			},
			"false",
		},
		"barExists": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"bothExist": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"barValid": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
					},
				},
			},
			"false",
		},
		"bothValid": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
						{
							Name: "bar",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
					},
				},
			},
			"true",
		},
	}
	// The point of this test is to verify that the below template works.
	tmpl := `{{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "foo") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`
	printer, err := NewGoTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}

	for name, item := range table {
		buffer := &bytes.Buffer{}
		err = printer.PrintObj(&item.pod, buffer)
		if err != nil {
			t.Errorf("%v: unexpected err: %v", name, err)
			continue
		}
		actual := buffer.String()
		if len(actual) == 0 {
			actual = "false"
		}
		if e := item.expect; e != actual {
			t.Errorf("%v: expected %v, got %v", name, e, actual)
		}
	}
}

func TestTemplatePanic(t *testing.T) {
	tmpl := `{{and ((index .currentState.info "foo").state.running.startedAt) .currentState.info.net.state.running.startedAt}}`
	printer, err := NewGoTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}
	buffer := &bytes.Buffer{}
	err = printer.PrintObj(&v1.Pod{}, buffer)
	if err == nil {
		t.Fatalf("expected that template to crash")
	}
	if buffer.String() == "" {
		t.Errorf("no debugging info was printed")
	}
}

func TestTemplateSuccess(t *testing.T) {

	templatePrinter, err := NewGoTemplatePrinter([]byte("{{.name}}"))
	if err != nil {
		t.Fatal(err)
	}

	// Template printer should succeed on these resources.
	om := func(name string) metav1.ObjectMeta { return metav1.ObjectMeta{Name: name} }
	objects := []runtime.Object{
		&v1.Pod{ObjectMeta: om("pod")},
		&v1.PodList{},
		&v1.PodList{Items: []v1.Pod{{}}},
		&v1.Endpoints{
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []v1.EndpointPort{{Port: 8080}},
			}}},
	}

	for _, obj := range objects {
		b := &bytes.Buffer{}
		if err := templatePrinter.PrintObj(obj, b); err != nil {
			t.Errorf("Unexpected template error: %v", err)
		}
	}
}

func TestTemplateErrors(t *testing.T) {

	templatePrinter, err := NewGoTemplatePrinter([]byte("{{len .items}}"))
	if err != nil {
		t.Fatal(err)
	}

	// Template printer should fail on these resources.
	om := func(name string) metav1.ObjectMeta { return metav1.ObjectMeta{Name: name} }
	objects := []runtime.Object{
		&v1.Pod{ObjectMeta: om("pod")},
		&v1.PodList{},
		&v1.Endpoints{
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []v1.EndpointPort{{Port: 8080}},
			}}},
	}

	for _, obj := range objects {
		b := &bytes.Buffer{}
		if err := templatePrinter.PrintObj(obj, b); err == nil {
			t.Errorf("Expected template printer error; received none")
		}
	}
}
