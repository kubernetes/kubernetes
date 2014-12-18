// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestExecCreate(t *testing.T) {
	jsonContainer := `{"Id": "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"}`
	var expected struct{ ID string }
	err := json.Unmarshal([]byte(jsonContainer), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonContainer, status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := CreateExecOptions{
		Container:    "test",
		AttachStdin:  true,
		AttachStdout: true,
		AttachStderr: false,
		Tty:          false,
		Cmd:          []string{"touch", "/tmp/file"},
	}
	execObj, err := client.CreateExec(config)
	if err != nil {
		t.Fatal(err)
	}
	expectedID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	if execObj.ID != expectedID {
		t.Errorf("ExecCreate: wrong ID. Want %q. Got %q.", expectedID, execObj.ID)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecCreate: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/containers/test/exec"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	var gotBody struct{ ID string }
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
}

func TestExecStartDetached(t *testing.T) {
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	fakeRT := &FakeRoundTripper{status: http.StatusOK}
	client := newTestClient(fakeRT)
	config := StartExecOptions{
		Detach: true,
	}
	err := client.StartExec(execID, config)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecStart: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/exec/" + execID + "/start"))
	if gotPath := req.URL.Path; gotPath != expectedURL.Path {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
	t.Log(req.Body)
	var gotBody struct{ Detach bool }
	err = json.NewDecoder(req.Body).Decode(&gotBody)
	if err != nil {
		t.Fatal(err)
	}
	if !gotBody.Detach {
		t.Fatal("Expected Detach in StartExecOptions to be true")
	}
}

func TestExecStartAndAttach(t *testing.T) {
	var reader = strings.NewReader("send value")
	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0, 0, 0, 0, 0, 0, 5})
		w.Write([]byte("hello"))
		req = *r
	}))
	defer server.Close()
	client, _ := NewClient(server.URL)
	client.SkipServerVersionCheck = true
	var stdout, stderr bytes.Buffer
	success := make(chan struct{})
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	opts := StartExecOptions{
		OutputStream: &stdout,
		ErrorStream:  &stderr,
		InputStream:  reader,
		RawTerminal:  true,
		Success:      success,
	}
	go client.StartExec(execID, opts)
	<-success
}

func TestExecResize(t *testing.T) {
	execID := "4fa6e0f0c6786287e131c3852c58a2e01cc697a68231826813597e4994f1d6e2"
	fakeRT := &FakeRoundTripper{status: http.StatusOK}
	client := newTestClient(fakeRT)
	err := client.ResizeExecTTY(execID, 10, 20)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	if req.Method != "POST" {
		t.Errorf("ExecStart: wrong HTTP method. Want %q. Got %q.", "POST", req.Method)
	}
	expectedURL, _ := url.Parse(client.getURL("/exec/" + execID + "/resize?h=10&w=20"))
	if gotPath := req.URL.RequestURI(); gotPath != expectedURL.RequestURI() {
		t.Errorf("ExecCreate: Wrong path in request. Want %q. Got %q.", expectedURL.Path, gotPath)
	}
}
