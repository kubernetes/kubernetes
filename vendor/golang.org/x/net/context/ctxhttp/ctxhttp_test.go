// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package ctxhttp

import (
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/net/context"
)

const (
	requestDuration = 100 * time.Millisecond
	requestBody     = "ok"
)

func okHandler(w http.ResponseWriter, r *http.Request) {
	time.Sleep(requestDuration)
	io.WriteString(w, requestBody)
}

func TestNoTimeout(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(okHandler))
	defer ts.Close()

	ctx := context.Background()
	res, err := Get(ctx, nil, ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != requestBody {
		t.Errorf("body = %q; want %q", slurp, requestBody)
	}
}

func TestCancelBeforeHeaders(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	blockServer := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cancel()
		<-blockServer
		io.WriteString(w, requestBody)
	}))
	defer ts.Close()
	defer close(blockServer)

	res, err := Get(ctx, nil, ts.URL)
	if err == nil {
		res.Body.Close()
		t.Fatal("Get returned unexpected nil error")
	}
	if err != context.Canceled {
		t.Errorf("err = %v; want %v", err, context.Canceled)
	}
}

func TestCancelAfterHangingRequest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-w.(http.CloseNotifier).CloseNotify()
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	resp, err := Get(ctx, nil, ts.URL)
	if err != nil {
		t.Fatalf("unexpected error in Get: %v", err)
	}

	// Cancel befer reading the body.
	// Reading Request.Body should fail, since the request was
	// canceled before anything was written.
	cancel()

	done := make(chan struct{})

	go func() {
		b, err := ioutil.ReadAll(resp.Body)
		if len(b) != 0 || err == nil {
			t.Errorf(`Read got (%q, %v); want ("", error)`, b, err)
		}
		close(done)
	}()

	select {
	case <-time.After(1 * time.Second):
		t.Errorf("Test timed out")
	case <-done:
	}
}
