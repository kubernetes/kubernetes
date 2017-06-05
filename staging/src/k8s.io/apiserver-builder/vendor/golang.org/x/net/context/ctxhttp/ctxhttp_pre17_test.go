// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!go1.7

package ctxhttp

import (
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
)

// golang.org/issue/14065
func TestClosesResponseBodyOnCancel(t *testing.T) {
	defer func() { testHookContextDoneBeforeHeaders = nop }()
	defer func() { testHookDoReturned = nop }()
	defer func() { testHookDidBodyClose = nop }()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())

	// closed when Do enters select case <-ctx.Done()
	enteredDonePath := make(chan struct{})

	testHookContextDoneBeforeHeaders = func() {
		close(enteredDonePath)
	}

	testHookDoReturned = func() {
		// We now have the result (the Flush'd headers) at least,
		// so we can cancel the request.
		cancel()

		// But block the client.Do goroutine from sending
		// until Do enters into the <-ctx.Done() path, since
		// otherwise if both channels are readable, select
		// picks a random one.
		<-enteredDonePath
	}

	sawBodyClose := make(chan struct{})
	testHookDidBodyClose = func() { close(sawBodyClose) }

	tr := &http.Transport{}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}
	req, _ := http.NewRequest("GET", ts.URL, nil)
	_, doErr := Do(ctx, c, req)

	select {
	case <-sawBodyClose:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for body to close")
	}

	if doErr != ctx.Err() {
		t.Errorf("Do error = %v; want %v", doErr, ctx.Err())
	}
}

type noteCloseConn struct {
	net.Conn
	onceClose sync.Once
	closefn   func()
}

func (c *noteCloseConn) Close() error {
	c.onceClose.Do(c.closefn)
	return c.Conn.Close()
}
