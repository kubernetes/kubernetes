/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestReplaceObject(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Printer = &testPrinter{}
	deleted := false
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodDelete:
				deleted = true
				fallthrough
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodGet:
				statusCode := http.StatusOK
				if deleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
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

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Printer = &testPrinter{}
	redisMasterDeleted := false
	frontendDeleted := false
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodDelete:
				redisMasterDeleted = true
				fallthrough
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodGet:
				statusCode := http.StatusOK
				if redisMasterDeleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodDelete:
				frontendDeleted = true
				fallthrough
			case p == "/namespaces/test/services/frontend" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodGet:
				statusCode := http.StatusOK
				if frontendDeleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
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
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Printer = &testPrinter{}
	created := map[string]bool{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodPut:
				created[p] = true
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodGet:
				statusCode := http.StatusNotFound
				if created[p] {
					statusCode = http.StatusOK
				}
				return &http.Response{StatusCode: statusCode, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodDelete:
				delete(created, p)
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers") && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy")
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\nreplicationcontroller/rc1\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/frontend\nreplicationcontroller/redis-master\nreplicationcontroller/redis-slave\n"+
		"replicationcontroller/rc1\nreplicationcontroller/rc1\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestForceReplaceObjectNotFound(t *testing.T) {
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && (m == http.MethodGet || m == http.MethodDelete):
				return &http.Response{StatusCode: http.StatusNotFound, Header: defaultHeader(), Body: stringBody("")}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdReplace(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
