// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package runtime

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/system"
)

func TestRunInBackgroundSendFirst(t *testing.T) { testRunInBackground(t, true) }
func TestRunInBackgroundRecvFirst(t *testing.T) { testRunInBackground(t, false) }

func testRunInBackground(t *testing.T, sendFirst bool) {
	srv := httptest.NewServer(nil)
	defer srv.Close()

	const id = "f00bar"
	sendWait, recvWait := make(chan bool), make(chan bool)
	sbr := make(chan bool) // strobed when system.StartBackgroundRequest has started

	calls := 0
	c := aetesting.FakeSingleContext(t, "system", "StartBackgroundRequest", func(req *pb.StartBackgroundRequestRequest, res *pb.StartBackgroundRequestResponse) error {
		calls++
		if calls > 1 {
			t.Errorf("Too many calls to system.StartBackgroundRequest")
		}
		sbr <- true
		res.RequestId = proto.String(id)
		<-sendWait
		return nil
	})

	var c2 context.Context // a fake
	newContext = func(*http.Request) context.Context {
		return c2
	}

	var fRun int
	f := func(c3 context.Context) {
		fRun++
		if c3 != c2 {
			t.Errorf("f got a different context than expected")
		}
	}

	ribErrc := make(chan error)
	go func() {
		ribErrc <- RunInBackground(c, f)
	}()

	brErrc := make(chan error)
	go func() {
		<-sbr
		req, err := http.NewRequest("GET", srv.URL+"/_ah/background", nil)
		if err != nil {
			brErrc <- fmt.Errorf("http.NewRequest: %v", err)
			return
		}
		req.Header.Set("X-AppEngine-BackgroundRequest", id)
		client := &http.Client{
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
			},
		}

		<-recvWait
		_, err = client.Do(req)
		brErrc <- err
	}()

	// Send and receive are both waiting at this point.
	waits := [2]chan bool{sendWait, recvWait}
	if !sendFirst {
		waits[0], waits[1] = waits[1], waits[0]
	}
	waits[0] <- true
	time.Sleep(100 * time.Millisecond)
	waits[1] <- true

	if err := <-ribErrc; err != nil {
		t.Fatalf("RunInBackground: %v", err)
	}
	if err := <-brErrc; err != nil {
		t.Fatalf("background request: %v", err)
	}

	if fRun != 1 {
		t.Errorf("Got %d runs of f, want 1", fRun)
	}
}
