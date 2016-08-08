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
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
)

func TestExtraArgsFail(t *testing.T) {
	initTestErrorHandler(t)
	buf := bytes.NewBuffer([]byte{})

	f, _, _, _ := NewAPIFactory()
	c := NewCmdCreate(f, buf)
	if ValidateArgs(c, []string{"rc"}) == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestCreateObject(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()
	rc.Items[0].Name = "redis-master-controller"

	f, tf, codec, _ := NewAPIFactory()
	contentConfig := dynamic.ContentConfig()
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/v1/namespaces/test/replicationcontrollers" && m == "POST":
			rw.WriteHeader(http.StatusCreated)
			codec.Encode(&rc.Items[0], rw)
		default:
			t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
		}
	}))
	defer srv.Close()
	contentConfig.GroupVersion = &unversioned.GroupVersion{Version: "v1"}

	cl, err := restclient.RESTClientFor(&restclient.Config{
		ContentConfig: contentConfig,
		Host:          srv.URL,
	})
	if err != nil {
		t.Fatalf("unable to create rest client: %s", err)
	}

	tf.Printer = &testPrinter{}
	tf.Client = cl
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateMultipleObject(t *testing.T) {
	initTestErrorHandler(t)
	_, svc, rc := testData()

	f, tf, codec, _ := NewAPIFactory()
	contentConfig := dynamic.ContentConfig()
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/v1/namespaces/test/services" && m == "POST":
			rw.WriteHeader(http.StatusCreated)
			codec.Encode(&svc.Items[0], rw)
		case p == "/v1/namespaces/test/replicationcontrollers" && m == "POST":
			rw.WriteHeader(http.StatusCreated)
			codec.Encode(&rc.Items[0], rw)
		default:
			t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
		}
	}))
	defer srv.Close()
	contentConfig.GroupVersion = &unversioned.GroupVersion{Version: "v1"}

	cl, err := restclient.RESTClientFor(&restclient.Config{
		ContentConfig: contentConfig,
		Host:          srv.URL,
	})
	if err != nil {
		t.Fatalf("unable to create rest client: %s", err)
	}

	tf.Printer = &testPrinter{}
	tf.Client = cl
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateDirectory(t *testing.T) {
	initTestErrorHandler(t)
	_, _, rc := testData()
	rc.Items[0].Name = "name"

	f, tf, codec, _ := NewAPIFactory()
	contentConfig := dynamic.ContentConfig()
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		switch p, m := req.URL.Path, req.Method; {
		case p == "/v1/namespaces/test/replicationcontrollers" && m == "POST":
			rw.WriteHeader(http.StatusCreated)
			codec.Encode(&rc.Items[0], rw)
		default:
			t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
		}
	}))
	defer srv.Close()
	contentConfig.GroupVersion = &unversioned.GroupVersion{Version: "v1"}

	cl, err := restclient.RESTClientFor(&restclient.Config{
		ContentConfig: contentConfig,
		Host:          srv.URL,
	})
	if err != nil {
		t.Fatalf("unable to create rest client: %s", err)
	}

	tf.Printer = &testPrinter{}
	tf.Client = cl
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreate(f, buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/legacy")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/name\nreplicationcontroller/name\nreplicationcontroller/name\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
