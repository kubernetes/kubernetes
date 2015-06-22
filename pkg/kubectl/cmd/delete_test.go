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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestDeleteObjectByTuple(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				// Ensures no GET is performed when deleting by name
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{"replicationcontrollers/redis-master-controller"})

	if buf.String() != "replicationcontrollers/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteNamedObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				// Ensures no GET is performed when deleting by name
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{"replicationcontrollers", "redis-master-controller"})

	if buf.String() != "replicationcontrollers/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontrollers/redis-master\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteObjectNotFound(t *testing.T) {
	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("cascade", "false")
	filenames := cmd.Flags().Lookup("filename").Value.(*util.StringList)
	err := RunDelete(f, buf, cmd, []string{}, *filenames)
	if err == nil || !errors.IsNotFound(err) {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}
}

func TestDeleteObjectIgnoreNotFound(t *testing.T) {
	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("ignore-not-found", "true")
	cmd.Run(cmd, []string{})

	if buf.String() != "" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObject(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontrollers/redis-master\nservices/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObjectContinueOnMissing(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: stringBody("")}, nil
			case p == "/namespaces/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("cascade", "false")
	filenames := cmd.Flags().Lookup("filename").Value.(*util.StringList)
	t.Logf("filenames: %v\n", filenames)
	err := RunDelete(f, buf, cmd, []string{}, *filenames)
	if err == nil || !errors.IsNotFound(err) {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}

	if buf.String() != "services/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteDirectory(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/namespaces/test/services/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontrollers/frontend\nservices/frontend\nreplicationcontrollers/redis-master\nservices/redis-master\nreplicationcontrollers/redis-slave\nservices/redis-slave\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleSelector(t *testing.T) {
	pods, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods" && m == "GET":
				if req.URL.Query().Get(api.LabelSelectorQueryParam(testapi.Version())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case p == "/namespaces/test/services" && m == "GET":
				if req.URL.Query().Get(api.LabelSelectorQueryParam(testapi.Version())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			case strings.HasPrefix(p, "/namespaces/test/pods/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/services/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf)
	cmd.Flags().Set("selector", "a=b")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{"pods,services"})

	if buf.String() != "pods/foo\npods/bar\nservices/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
