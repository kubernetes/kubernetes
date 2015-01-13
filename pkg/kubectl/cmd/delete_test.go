/*
Copyright 2014 Google Inc. All rights reserved.

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

package cmd_test

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func TestDeleteObject(t *testing.T) {
	pods, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master.json")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "redis-master\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteObjectIgnoreNotFound(t *testing.T) {
	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master.json")
	cmd.Run(cmd, []string{})

	if buf.String() != "" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObject(t *testing.T) {
	pods, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			case p == "/ns/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master.json")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.json")
	cmd.Run(cmd, []string{})

	if buf.String() != "redis-master\nfrontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObjectIgnoreMissing(t *testing.T) {
	_, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
			case p == "/ns/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master.json")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.json")
	cmd.Run(cmd, []string{})

	if buf.String() != "frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteDirectory(t *testing.T) {
	pods, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/ns/test/pods/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			case strings.HasPrefix(p, "/ns/test/services/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case strings.HasPrefix(p, "/ns/test/replicationcontrollers/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("filename", "../../../examples/guestbook")
	cmd.Run(cmd, []string{})

	if buf.String() != "frontendController\nfrontend\nredis-master\nredis-master\nredisSlaveController\nredisslave\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleSelector(t *testing.T) {
	pods, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods" && m == "GET":
				if req.URL.Query().Get("labels") != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case p == "/ns/test/services" && m == "GET":
				if req.URL.Query().Get("labels") != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			case strings.HasPrefix(p, "/ns/test/pods/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			case strings.HasPrefix(p, "/ns/test/services/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdDelete(buf)
	cmd.Flags().String("namespace", "test", "")
	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods,services"})

	if buf.String() != "foo\nbar\nbaz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
