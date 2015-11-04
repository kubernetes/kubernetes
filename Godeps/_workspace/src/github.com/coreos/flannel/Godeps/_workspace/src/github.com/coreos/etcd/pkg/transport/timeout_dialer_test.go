// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"net"
	"testing"
	"time"
)

func TestReadWriteTimeoutDialer(t *testing.T) {
	stop := make(chan struct{})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("unexpected listen error: %v", err)
	}
	ts := testBlockingServer{ln, 2, stop}
	go ts.Start(t)

	d := rwTimeoutDialer{
		wtimeoutd:  10 * time.Millisecond,
		rdtimeoutd: 10 * time.Millisecond,
	}
	conn, err := d.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("unexpected dial error: %v", err)
	}
	defer conn.Close()

	// fill the socket buffer
	data := make([]byte, 5*1024*1024)
	timer := time.AfterFunc(d.wtimeoutd*5, func() {
		t.Fatal("wait timeout")
	})
	defer timer.Stop()

	_, err = conn.Write(data)
	if operr, ok := err.(*net.OpError); !ok || operr.Op != "write" || !operr.Timeout() {
		t.Errorf("err = %v, want write i/o timeout error", err)
	}

	timer.Reset(d.rdtimeoutd * 5)

	conn, err = d.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("unexpected dial error: %v", err)
	}
	defer conn.Close()

	buf := make([]byte, 10)
	_, err = conn.Read(buf)
	if operr, ok := err.(*net.OpError); !ok || operr.Op != "read" || !operr.Timeout() {
		t.Errorf("err = %v, want write i/o timeout error", err)
	}

	stop <- struct{}{}
}

type testBlockingServer struct {
	ln   net.Listener
	n    int
	stop chan struct{}
}

func (ts *testBlockingServer) Start(t *testing.T) {
	for i := 0; i < ts.n; i++ {
		conn, err := ts.ln.Accept()
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Close()
	}
	<-ts.stop
}
