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

// TestNewTimeoutListener tests that NewTimeoutListener returns a
// rwTimeoutListener struct with timeouts set.
func TestNewTimeoutListener(t *testing.T) {
	l, err := NewTimeoutListener("127.0.0.1:0", "http", TLSInfo{}, time.Hour, time.Hour)
	if err != nil {
		t.Fatalf("unexpected NewTimeoutListener error: %v", err)
	}
	defer l.Close()
	tln := l.(*rwTimeoutListener)
	if tln.rdtimeoutd != time.Hour {
		t.Errorf("read timeout = %s, want %s", tln.rdtimeoutd, time.Hour)
	}
	if tln.wtimeoutd != time.Hour {
		t.Errorf("write timeout = %s, want %s", tln.wtimeoutd, time.Hour)
	}
}

func TestWriteReadTimeoutListener(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("unexpected listen error: %v", err)
	}
	wln := rwTimeoutListener{
		Listener:   ln,
		wtimeoutd:  10 * time.Millisecond,
		rdtimeoutd: 10 * time.Millisecond,
	}
	stop := make(chan struct{})

	blocker := func() {
		conn, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			t.Fatalf("unexpected dail error: %v", err)
		}
		defer conn.Close()
		// block the receiver until the writer timeout
		<-stop
	}
	go blocker()

	conn, err := wln.Accept()
	if err != nil {
		t.Fatalf("unexpected accept error: %v", err)
	}
	defer conn.Close()

	// fill the socket buffer
	data := make([]byte, 5*1024*1024)
	timer := time.AfterFunc(wln.wtimeoutd*5, func() {
		t.Fatal("wait timeout")
	})
	defer timer.Stop()

	_, err = conn.Write(data)
	if operr, ok := err.(*net.OpError); !ok || operr.Op != "write" || !operr.Timeout() {
		t.Errorf("err = %v, want write i/o timeout error", err)
	}
	stop <- struct{}{}

	timer.Reset(wln.rdtimeoutd * 5)
	go blocker()

	conn, err = wln.Accept()
	if err != nil {
		t.Fatalf("unexpected accept error: %v", err)
	}
	buf := make([]byte, 10)
	_, err = conn.Read(buf)
	if operr, ok := err.(*net.OpError); !ok || operr.Op != "read" || !operr.Timeout() {
		t.Errorf("err = %v, want write i/o timeout error", err)
	}
	stop <- struct{}{}
}
