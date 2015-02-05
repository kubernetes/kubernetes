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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func TestCreateObject(t *testing.T) {
	pods, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &pods.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdCreate(buf)
	cmd.Flags().Set("filename", "../../../test/api-resources/test-pod.json")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "test-pod\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateMultipleObject(t *testing.T) {
	pods, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &pods.Items[0])}, nil
			case p == "/ns/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdCreate(buf)
	cmd.Flags().Set("filename", "../../../test/api-resources/test-pod.json")
	cmd.Flags().Set("filename", "../../../test/api-resources/test-service.json")
	cmd.Run(cmd, []string{})

	if buf.String() != "test-pod\ntest-service\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateDirectory(t *testing.T) {
	pods, svc := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/ns/test/pods" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &pods.Items[0])}, nil
			case p == "/ns/test/services" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			case p == "/ns/test/replicationcontrollers" && m == "POST":
				return &http.Response{StatusCode: 201, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := f.NewCmdCreate(buf)
	cmd.Flags().Set("filename", "../../../test/api-resources")
	cmd.Run(cmd, []string{})

	if buf.String() != "test-controller\ntest-pod\ntest-service\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
