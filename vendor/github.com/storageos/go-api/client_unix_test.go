// +build !windows
// Copyright 2016 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package storageos

import (
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

const (
	nativeProtocol     = unixProtocol
	nativeRealEndpoint = "unix:///var/run/docker.sock"
	nativeBadEndpoint  = "unix:///tmp/echo.sock"
)

func TestNewTSLAPIClientUnixEndpoint(t *testing.T) {
	srv, cleanup, err := newNativeServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	srv.Start()
	defer srv.Close()
	endpoint := nativeProtocol + "://" + srv.Listener.Addr().String()
	client, err := newTLSClient(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	rsp, err := client.do("GET", "/", doOptions{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := ioutil.ReadAll(rsp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "ok" {
		t.Fatalf("Expected response to be %q, got: %q", "ok", string(data))
	}
}

func newNativeServer(handler http.Handler) (*httptest.Server, func(), error) {
	tmpdir, err := ioutil.TempDir("", "socket")
	if err != nil {
		return nil, nil, err
	}
	socketPath := filepath.Join(tmpdir, "docker_test_stress.sock")
	l, err := net.Listen("unix", socketPath)
	if err != nil {
		return nil, nil, err
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = l
	return srv, func() { os.RemoveAll(tmpdir) }, nil
}
