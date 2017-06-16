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

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
var falseVar = false

func TestDeleteObjectByTuple(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, o := req.URL.Path, req.Method, parseDeleteOptions(req.Body); {

			// replication controller with cascade off
			// Return 200 for first DELETE request.
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE" && o.Preconditions == nil:
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
				// Return 404 for subsequent DELETE requests with precondition.
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE" && o.Preconditions != nil:
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

				// secret with cascade on, but no client-side reaper.
				// Return 202 for first DELETE request.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && o.Preconditions == nil:
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
				// Return 404 for subsequent DELETE requests with precondition.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && o.Preconditions != nil:
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

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
	cmd.Flags().Set("timeout", "1m")
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

func parseDeleteOptions(body io.ReadCloser) *metav1.DeleteOptions {
	if body == nil {
		return nil
	}
	var parsedBody metav1.DeleteOptions
	rawBody, _ := ioutil.ReadAll(body)
	json.Unmarshal(rawBody, &parsedBody)
	return &parsedBody
}

func hasExpectedOrphanDependents(options *metav1.DeleteOptions, expectedOrphanDependents *bool) bool {
	if options == nil || expectedOrphanDependents == nil {
		return options == nil && expectedOrphanDependents == nil
	}
	if options.OrphanDependents == nil {
		return false
	}
	return *expectedOrphanDependents == *options.OrphanDependents
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
			switch p, m, o := req.URL.Path, req.Method, parseDeleteOptions(req.Body); {
			// Return 200 for first DELETE request.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && hasExpectedOrphanDependents(o, expectedOrphanDependents) && o.Preconditions == nil:
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

				// Return 404 for subsequent requests with preconditions.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && hasExpectedOrphanDependents(o, expectedOrphanDependents) && o.Preconditions != nil:
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

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
			switch p, m, o := req.URL.Path, req.Method, parseDeleteOptions(req.Body); {

			// replication controller with cascade off.
			// Return 200 for first DELETE request.
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE" && o.Preconditions == nil:
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
				// Return 404 for subsequent requests with preconditions.
			case p == "/namespaces/test/replicationcontrollers/redis-master-controller" && m == "DELETE" && o.Preconditions != nil:
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

			// secret with cascade on, but no client-side reaper.
			// Return 202 for first DELETE request.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && o.Preconditions == nil:
				return &http.Response{StatusCode: 202, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			// Return 404 for subsequent DELETE requests with preconditions.
			case p == "/namespaces/test/secrets/mysecret" && m == "DELETE" && o.Preconditions != nil:
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil

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
	cmd.Flags().Set("wait", "false")
	cmd.Flags().Set("timeout", "1m")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestWaitForDelete(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	requests := []*http.Request{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch m := req.Method; {
			case m == "DELETE" && len(requests) < 2:
				requests = append(requests, req)
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			case m == "DELETE" && len(requests) == 2:
				requests = append(requests, req)
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	trueVar := true
	falseVar := false

	testCases := []struct {
		name                 string
		cascade              *bool
		wait                 *bool
		expectedOutput       string
		expectedNumOfDelReqs int
	}{
		{
			// No wait when wait is false.
			"secret/mysecret",
			&falseVar,
			&falseVar,
			"secret/mysecret\n",
			1,
		},
		{
			// Waits when cascade is false.
			"secret/mysecret",
			&falseVar,
			nil,
			"secret/mysecret\n",
			3,
		},
		{
			// No wait when wait is false.
			"secret/mysecret",
			nil,
			&falseVar,
			"secret/mysecret\n",
			1,
		},
		{
			// Waits when both wait and cascade are true.
			"secret/mysecret",
			&trueVar,
			&trueVar,
			"secret/mysecret\n",
			3,
		},
		{
			// Should not wait for ns deletion by default.
			"namespaces/myns",
			nil,
			nil,
			"namespace/myns\n",
			1,
		},
		{
			// Should wait for ns deletion when --wait is set to true.
			"namespaces/myns",
			nil,
			&trueVar,
			"namespace/myns\n",
			3,
		},
	}
	for i, c := range testCases {
		requests = []*http.Request{}
		buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})
		cmd := NewCmdDelete(f, buf, errBuf)
		cmd.Flags().Set("namespace", "test")
		setFlag(cmd, "wait", c.wait)
		setFlag(cmd, "cascade", c.cascade)
		cmd.Flags().Set("timeout", "1m")
		cmd.Flags().Set("output", "name")
		cmd.Run(cmd, []string{c.name})

		if buf.String() != c.expectedOutput {
			t.Errorf("%d: unexpected output: %s, expected: %s", i, buf.String(), c.expectedOutput)
		}
		if c.expectedNumOfDelReqs != len(requests) {
			t.Errorf("%d: unexpected number of DELETE requests: %d, expected: %d", i, len(requests), c.expectedNumOfDelReqs)
		}
	}
}

func setFlag(cmd *cobra.Command, flagName string, value *bool) {
	if value == nil {
		return
	}
	switch *value {
	case true:
		cmd.Flags().Set(flagName, "true")
	case false:
		cmd.Flags().Set(flagName, "false")
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
		WaitForDeletion: &falseVar,
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
	cmd.Flags().Set("wait", "false")
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
	cmd.Flags().Set("wait", "false")
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
		GracePeriod:     -1,
		Cascade:         false,
		WaitForDeletion: &falseVar,
		Output:          "name",
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
	cmd.Flags().Set("wait", "false")
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
	cmd.Flags().Set("wait", "false")
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
	cmd.Flags().Set("wait", "false")
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
			WaitForDeletion: &falseVar,
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

func TestGetUID(t *testing.T) {
	_, svcItems, _ := testData()
	svc := &svcItems.Items[0]
	statusUID := "status-uid"
	status := &metav1.Status{
		Details: &metav1.StatusDetails{
			UID: types.UID(statusUID),
		},
	}
	testCases := []struct {
		obj         runtime.Object
		expectedUID string
		expectedErr bool
	}{
		{
			svc,
			string(svc.ObjectMeta.UID),
			false,
		},
		{
			status,
			statusUID,
			false,
		},
		{
			nil,
			"",
			true,
		},
		{
			// No error for backward compatibility.
			// TODO: Update this to return an error in 1.8.
			&metav1.Status{},
			"",
			false,
		},
	}
	for i, c := range testCases {
		uid, err := getUID(c.obj)
		if c.expectedErr == true && err == nil || c.expectedErr == false && err != nil {
			t.Errorf("%d: Unexpected err: %s, expected err: %t", i, err, c.expectedErr)
		}
		if !c.expectedErr && uid != c.expectedUID {
			t.Errorf("%d: Unexpected uid: %s, expected uid: %s", i, uid, c.expectedUID)
		}
	}
}
