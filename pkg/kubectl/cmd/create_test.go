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

package cmd

import (
	"bytes"
	"net/http"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestExtraArgsFail(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})

	f, _, _ := NewAPIFactory()
	c := NewCmdCreate(f, buf)
	if ValidateArgs(c, []string{"rc"}) == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestCreateObject(t *testing.T) {
	_, _, rc := testData()
	rc.Items[0].Name = "redis-master-controller"

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateMultipleObject(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateDirectory(t *testing.T) {
	_, svc, rc := testData()
	rc.Items[0].Name = "name"

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/name\nservice/baz\nreplicationcontroller/name\nservice/baz\nreplicationcontroller/name\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPrintObjectSpecificMessage(t *testing.T) {
	tests := []struct {
		obj          runtime.Object
		expectOutput bool
	}{
		{
			obj:          &api.Service{},
			expectOutput: false,
		},
		{
			obj:          &api.Pod{},
			expectOutput: false,
		},
		{
			obj:          &api.Service{Spec: api.ServiceSpec{Type: api.ServiceTypeLoadBalancer}},
			expectOutput: false,
		},
		{
			obj:          &api.Service{Spec: api.ServiceSpec{Type: api.ServiceTypeNodePort}},
			expectOutput: true,
		},
	}
	for _, test := range tests {
		buff := &bytes.Buffer{}
		printObjectSpecificMessage(test.obj, buff)
		if test.expectOutput && buff.Len() == 0 {
			t.Errorf("Expected output, saw none for %v", test.obj)
		}
		if !test.expectOutput && buff.Len() > 0 {
			t.Errorf("Expected no output, saw %s for %v", buff.String(), test.obj)
		}
	}
}

func TestMakePortsString(t *testing.T) {
	tests := []struct {
		ports          []api.ServicePort
		useNodePort    bool
		expectedOutput string
	}{
		{ports: nil, expectedOutput: ""},
		{ports: []api.ServicePort{}, expectedOutput: ""},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				Protocol: "UDP",
			},
			{
				Port:     9000,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80,udp:8080,tcp:9000",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				NodePort: 9090,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				NodePort: 80,
				Protocol: "UDP",
			},
		},
			useNodePort:    true,
			expectedOutput: "tcp:9090,udp:80",
		},
	}
	for _, test := range tests {
		output := makePortsString(test.ports, test.useNodePort)
		if output != test.expectedOutput {
			t.Errorf("expected: %s, saw: %s.", test.expectedOutput, output)
		}
	}
}
