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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
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
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.json")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontrollers/redis-master-controller\n" {
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
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.json")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.json")
	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	if buf.String() != "replicationcontrollers/rc1\nservices/baz\n" {
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
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontrollers/name\nservices/baz\nreplicationcontrollers/name\nservices/baz\nreplicationcontrollers/name\nservices/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
