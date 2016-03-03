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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
)

func TestDeleteObjectByTuple(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers/redis-master-controller"})

	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteNamedObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers", "redis-master-controller"})

	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

type fakeResponseFunc func(req *http.Request) (*http.Response, error)

func TestDeleteObjectNotFound(t *testing.T) {
	_, _, rc := testData()
	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"

	notFoundError := &errors.NewNotFound(api.Resource("replicationcontrollers"), "redis-master").(*errors.StatusError).ErrStatus

	tests := []struct {
		Name             string
		Func             fakeResponseFunc
		ExpectNotFound   bool
		ExpectOtherError bool
	}{
		{
			"GET succeeds, DELETE fails with 400 BadRequest error",
			func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
					return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
					return &http.Response{StatusCode: 400, Body: stringBody("this is a bad request")}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			},
			false,
			true,
		},
		{
			"GET succeeds, DELETE fails with a 404 NotFound error",
			func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
					return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
					return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			},
			false,
			false,
		},
		{
			"GET fails with a 404 NotFound error, DELETE shouldn't matter",
			func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
					return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
				case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
					return &http.Response{StatusCode: 200, Body: stringBody("this shoudn't matter")}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			},
			true,
			false,
		},
	}

	for _, test := range tests {
		tf.Client = &fake.RESTClient{
			Codec:  codec,
			Client: fake.CreateHTTPClient(test.Func),
		}
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdDelete(f, buf)
		options := &DeleteOptions{
			Filenames: []string{"../../../examples/guestbook/redis-master-controller.yaml"},
		}
		cmd.Flags().Set("cascade", "false")
		cmd.Flags().Set("output", "name")
		err := RunDelete(f, buf, cmd, []string{}, options)
		if test.ExpectNotFound && (err == nil || !strings.Contains(err.Error(), "not found")) {
			t.Errorf("case: %q, expect error NotFound, got %v", test.Name, err)
		}
		if test.ExpectOtherError && (err == nil || errors.IsNotFound(err)) {
			t.Errorf("case: %q, expect an error that is not NotFound, got: %v", test.Name, err)
		}
		if !test.ExpectNotFound && !test.ExpectOtherError && err != nil {
			t.Errorf("case: %q, unexpected error: %v", test.Name, err)
		}
	}
}

func TestDeleteObjectIgnoreNotFound(t *testing.T) {
	notFoundError := &errors.NewNotFound(api.Resource("replicationcontrollers"), "redis-master").(*errors.StatusError).ErrStatus

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
				return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "No resources found\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteAllAlwaysIgnoreNotFound(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()

	// Add an item to the list which will result in a 404 on delete
	svc.Items = append(svc.Items, api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	notFoundError := &errors.NewNotFound(api.Resource("services"), "foo").(*errors.StatusError).ErrStatus

	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "GET":
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			case p == "/namespaces/test/services/foo" && m == "DELETE":
				return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
			case p == "/namespaces/test/services/baz" && m == "DELETE":
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
	cmd.Flags().Set("all", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	err := RunDelete(f, buf, cmd, []string{"services"}, &DeleteOptions{})
	if err != nil {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}
	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	// With --all, even if the --ignore-not-found is explicitly set to false,
	// NotFound errors are ignored.
	buf.Reset()
	cmd.Flags().Set("ignore-not-found", "false")
	err = RunDelete(f, buf, cmd, []string{"services"}, &DeleteOptions{})
	if err != nil {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}
	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObject(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && (m == "DELETE" || m == "GET"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && (m == "DELETE" || m == "GET"):
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nservice/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObjectContinueOnMissing(t *testing.T) {
	_, svc, _ := testData()
	notFoundError := &errors.NewNotFound(api.Resource("replicationcontrollers"), "redis-master").(*errors.StatusError).ErrStatus

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "GET":
				return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
			case p == "/namespaces/test/services/frontend" && (m == "DELETE" || m == "GET"):
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
	options := &DeleteOptions{
		Filenames: []string{"../../../examples/guestbook/redis-master-controller.yaml", "../../../examples/guestbook/frontend-service.yaml"},
	}
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	err := RunDelete(f, buf, cmd, []string{}, options)
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}

	if buf.String() != "service/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleResourcesWithTheSameName(t *testing.T) {
	_, svc, rc := testData()
	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/foo" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/foo" && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers,services", "baz", "foo"})
	if buf.String() != "replicationcontroller/baz\nreplicationcontroller/foo\nservice/baz\nservice/foo\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteDirectory(t *testing.T) {
	_, svc, rc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/namespaces/test/services/") && (m == "DELETE" || m == "GET"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && (m == "DELETE" || m == "GET"):
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/frontend\nservice/frontend\nreplicationcontroller/redis-master\nservice/redis-master\nreplicationcontroller/redis-slave\nservice/redis-slave\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleSelector(t *testing.T) {
	pods, svc, _ := testData()
	notFoundError := &errors.NewNotFound(api.Resource("pods"), "bar").(*errors.StatusError).ErrStatus

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods" && m == "GET":
				if req.URL.Query().Get(unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case p == "/namespaces/test/services" && m == "GET":
				if req.URL.Query().Get(unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			case strings.HasPrefix(p, "/namespaces/test/pods/foo") && m == "DELETE":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/pods/bar") && m == "DELETE":
				// When using a label selector to delete pods, NotFound error
				// during delete should be ignored.
				return &http.Response{StatusCode: 404, Body: objBody(codec, notFoundError)}, nil
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
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"pods,services"})

	if buf.String() != "pod/foo\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
