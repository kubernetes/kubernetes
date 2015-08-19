/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"

	client "k8s.io/kubernetes/pkg/client/unversioned"
)

func TestReplaceObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && (m == "GET" || m == "PUT" || m == "DELETE"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
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

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestReplaceMultipleObject(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && (m == "GET" || m == "PUT" || m == "DELETE"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && (m == "GET" || m == "PUT" || m == "DELETE"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nservice/frontend\nreplicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestReplaceDirectory(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/namespaces/test/services/") && (m == "GET" || m == "PUT" || m == "DELETE"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && (m == "GET" || m == "PUT" || m == "DELETE"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/services") && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers") && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook")
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\nservice/baz\nreplicationcontroller/rc1\nservice/baz\nreplicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/frontend\nservice/frontend\nreplicationcontroller/redis-master\nservice/redis-master\nreplicationcontroller/redis-slave\nservice/redis-slave\n"+
		"replicationcontroller/rc1\nservice/baz\nreplicationcontroller/rc1\nservice/baz\nreplicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestForceReplaceObjectNotFound(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
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

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
