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

package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

var unstructuredSerializer = dynamic.ContentConfig().NegotiatedSerializer

func TestDeleteObjectByTuple(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {

			// replication controller with cascade off
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

			// secret with cascade on, but no client-side reaper
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

			default:
				// Ensures no GET is performed when deleting by name
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"

	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers/redis-master-controller"})
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	// Test cascading delete of object without client-side reaper doesn't make GET requests
	buf, errBuf = bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd = NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"secrets/mysecret"})
	if buf.String() != "secret/mysecret\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func hasExpectedOrphanDependents(body io.ReadCloser, expectedOrphanDependents *bool) bool {
	if body == nil || expectedOrphanDependents == nil {
		return body == nil && expectedOrphanDependents == nil
	}
	var parsedBody metav1.DeleteOptions
	rawBody, _ := ioutil.ReadAll(body)
	json.Unmarshal(rawBody, &parsedBody)
	if parsedBody.OrphanDependents == nil {
		return false
	}
	return *expectedOrphanDependents == *parsedBody.OrphanDependents
}

// Tests that DeleteOptions.OrphanDependents is appropriately set while deleting objects.
func TestOrphanDependentsInDeleteObject(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	var expectedOrphanDependents *bool
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, b := req.URL.Path, req.Method, req.Body; {

			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && hasExpectedOrphanDependents(b, expectedOrphanDependents):

				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"

	// DeleteOptions.OrphanDependents should be false, when cascade is true (default).
	falseVar := false
	expectedOrphanDependents = &falseVar
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"secrets/mysecret"})
	if buf.String() != "secret/mysecret\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	// Test that delete options should be set to orphan when cascade is false.
	trueVar := true
	expectedOrphanDependents = &trueVar
	buf, errBuf = bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd = NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"secrets/mysecret"})
	if buf.String() != "secret/mysecret\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteNamedObject(t *testing.T) {
	initTestErrorHandler(t)
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {

			// replication controller with cascade off
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

			// secret with cascade on, but no client-side reaper
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

			default:
				// Ensures no GET is performed when deleting by name
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"

	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers", "redis-master-controller"})
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	// Test cascading delete of object without client-side reaper doesn't make GET requests
	buf, errBuf = bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
	cmd = NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"secrets", "mysecret"})
	if buf.String() != "secret/mysecret\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteObject(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

type fakeReaper struct {
	namespace, name string
	timeout         time.Duration
	deleteOptions   *metav1.DeleteOptions
	err             error
}

func (r *fakeReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *metav1.DeleteOptions) error {
	r.namespace, r.name = namespace, name
	r.timeout = timeout
	r.deleteOptions = gracePeriod
	return r.err
}

type fakeReaperFactory struct {
	cmdutil.Factory
	reaper kubectl.Reaper
}

func (f *fakeReaperFactory) Reaper(mapping *meta.RESTMapping) (kubectl.Reaper, error) {
	return f.reaper, nil
}

func TestDeleteObjectGraceZero(t *testing.T) {
	initTestErrorHandler(t)
	pods, _, _ := testData()

	objectDeletionWaitInterval = time.Millisecond
	count := 0
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Logf("got request %s %s", req.Method, req.URL.Path)
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods/nginx" && m == "GET":
				count++
				switch count {
				case 1, 2, 3:
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])}, nil
				default:
					return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &metav1.Status{})}, nil
				}
			case p == "/api/v1/namespaces/test" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
			case p == "/namespaces/test/pods/nginx" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	reaper := &fakeReaper{}
	fake := &fakeReaperFactory{Factory: f, reaper: reaper}
	cmd := NewCmdDelete(fake, buf, errBuf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("grace-period", "0")
	cmd.Run(cmd, []string{"pod/nginx"})

	// uses the name from the file, not the response
	if buf.String() != "pod/nginx\n" {
		t.Errorf("unexpected output: %s\n---\n%s", buf.String(), errBuf.String())
	}
	if reaper.deleteOptions == nil || reaper.deleteOptions.GracePeriodSeconds == nil || *reaper.deleteOptions.GracePeriodSeconds != 1 {
		t.Errorf("unexpected reaper options: %#v", reaper)
	}
	if count != 4 {
		t.Errorf("unexpected calls to GET: %d", count)
	}
}

func TestDeleteObjectNotFound(t *testing.T) {
	initTestErrorHandler(t)
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: stringBody("")}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	options := &DeleteOptions{
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../examples/guestbook/legacy/redis-master-controller.yaml"},
		},
		GracePeriod: -1,
		Cascade:     false,
		Output:      "name",
	}
	err := options.Complete(f, buf, errBuf, []string{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = options.RunDelete()
	if err == nil || !errors.IsNotFound(err) {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}
}

func TestDeleteObjectIgnoreNotFound(t *testing.T) {
	initTestErrorHandler(t)
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: stringBody("")}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("ignore-not-found", "true")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteAllNotFound(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, _ := testData()
	// Add an item to the list which will result in a 404 on delete
	svc.Items = append(svc.Items, api.Service{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	notFoundError := &errors.NewNotFound(api.Resource("services"), "foo").ErrStatus

	f, tf, codec, _ := cmdtesting.NewAPIFactory()

	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			case p == "/namespaces/test/services/foo" && m == "DELETE":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, notFoundError)}, nil
			case p == "/namespaces/test/services/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	// Make sure we can explicitly choose to fail on NotFound errors, even with --all
	options := &DeleteOptions{
		FilenameOptions: resource.FilenameOptions{},
		GracePeriod:     -1,
		Cascade:         false,
		DeleteAll:       true,
		IgnoreNotFound:  false,
		Output:          "name",
	}
	err := options.Complete(f, buf, errBuf, []string{"services"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = options.RunDelete()
	if err == nil || !errors.IsNotFound(err) {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}
}

func TestDeleteAllIgnoreNotFound(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, _ := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()

	// Add an item to the list which will result in a 404 on delete
	svc.Items = append(svc.Items, api.Service{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	notFoundError := &errors.NewNotFound(api.Resource("services"), "foo").ErrStatus

	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			case p == "/namespaces/test/services/foo" && m == "DELETE":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, notFoundError)}, nil
			case p == "/namespaces/test/services/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("all", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"services"})

	if buf.String() != "service/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObject(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nservice/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleObjectContinueOnMissing(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, _ := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == "DELETE":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: stringBody("")}, nil
			case p == "/namespaces/test/services/frontend" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	options := &DeleteOptions{
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../examples/guestbook/legacy/redis-master-controller.yaml", "../../../examples/guestbook/frontend-service.yaml"},
		},
		GracePeriod: -1,
		Cascade:     false,
		Output:      "name",
	}
	err := options.Complete(f, buf, errBuf, []string{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = options.RunDelete()
	if err == nil || !errors.IsNotFound(err) {
		t.Errorf("unexpected error: expected NotFound, got %v", err)
	}

	if buf.String() != "service/frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleResourcesWithTheSameName(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, rc := testData()
	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/foo" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/baz" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/foo" && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				// Ensures no GET is performed when deleting by name
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"replicationcontrollers,services", "baz", "foo"})
	if buf.String() != "replicationcontroller/baz\nreplicationcontroller/foo\nservice/baz\nservice/foo\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteDirectory(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/frontend\nreplicationcontroller/redis-master\nreplicationcontroller/redis-slave\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestDeleteMultipleSelector(t *testing.T) {
	initTestErrorHandler(t)
	pods, svc, _ := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods" && m == "GET":
				if req.URL.Query().Get(metav1.LabelSelectorQueryParam(api.Registry.GroupOrDie(api.GroupName).GroupVersion.String())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)}, nil
			case p == "/namespaces/test/services" && m == "GET":
				if req.URL.Query().Get(metav1.LabelSelectorQueryParam(api.Registry.GroupOrDie(api.GroupName).GroupVersion.String())) != "a=b" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			case strings.HasPrefix(p, "/namespaces/test/pods/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/services/") && m == "DELETE":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	cmd := NewCmdDelete(f, buf, errBuf)
	cmd.Flags().Set("selector", "a=b")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"pods,services"})

	if buf.String() != "pod/foo\npod/bar\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestResourceErrors(t *testing.T) {
	initTestErrorHandler(t)
	testCases := map[string]struct {
		args  []string
		errFn func(error) bool
	}{
		"no args": {
			args:  []string{},
			errFn: func(err error) bool { return strings.Contains(err.Error(), "You must provide one or more resources") },
		},
		"resources but no selectors": {
			args: []string{"pods"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), "resource(s) were provided, but no name, label selector, or --all flag specified")
			},
		},
		"multiple resources but no selectors": {
			args: []string{"pods,deployments"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), "resource(s) were provided, but no name, label selector, or --all flag specified")
			},
		},
	}

	for k, testCase := range testCases {
		f, tf, _, _ := cmdtesting.NewAPIFactory()
		tf.Printer = &testPrinter{}
		tf.Namespace = "test"
		tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion}}

		buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

		options := &DeleteOptions{
			FilenameOptions: resource.FilenameOptions{},
			GracePeriod:     -1,
			Cascade:         false,
			Output:          "name",
		}
		err := options.Complete(f, buf, errBuf, testCase.args)
		if !testCase.errFn(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}

		if tf.Printer.(*testPrinter).Objects != nil {
			t.Errorf("unexpected print to default printer")
		}
		if buf.Len() > 0 {
			t.Errorf("buffer should be empty: %s", string(buf.Bytes()))
		}
	}
}
