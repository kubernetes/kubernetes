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
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
)

func TestPatchObject(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{"services/frontend"})

	// uses the name from the file, not the response
	if buf.String() != "frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchObjectFromFile(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("filename", "../../../examples/guestbook/frontend-service.yaml")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "frontend\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchObjectWithConflictsRetriesSuccess(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	calls := 0
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				if m == "PATCH" {
					calls++
					if calls == 1 {
						return &http.Response{StatusCode: http.StatusConflict, Body: objBody(codec, &unversioned.Status{})}, nil
					}
					return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
				}
				return &http.Response{StatusCode: http.StatusOK, Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("max-retries", "2")
	err := RunPatch(f, buf, cmd, []string{"services/frontend"}, false, &PatchOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Errorf("expected: %d calls, saw %d", 2, calls)
	}
	// uses the name from the file, not the response
	if buf.String() != "\"frontend\" patched\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPatchObjectWithConflictsTillGiveUp(t *testing.T) {
	_, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	calls := 0
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services/frontend" && (m == "PATCH" || m == "GET"):
				if m == "PATCH" {
					calls++
					return &http.Response{StatusCode: http.StatusConflict, Body: objBody(codec, &unversioned.Status{})}, nil
				}
				return &http.Response{StatusCode: http.StatusOK, Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdPatch(f, buf)
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("patch", `{"spec":{"type":"NodePort"}}`)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("max-retries", "2")
	err := RunPatch(f, buf, cmd, []string{"services/frontend"}, false, &PatchOptions{})
	if err == nil {
		t.Error("unexpected non-error")
	}
	if calls != 2 {
		t.Errorf("expected: %d calls, saw %d", 2, calls)
	}
	if len(buf.String()) != 0 {
		t.Errorf("expected empty output, saw: '%s'", buf.String())
	}
}
