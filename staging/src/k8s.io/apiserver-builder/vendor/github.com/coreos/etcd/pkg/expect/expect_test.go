// Copyright 2016 The etcd Authors
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

// build !windows

package expect

import (
	"os"
	"testing"
	"time"
)

func TestExpectFunc(t *testing.T) {
	ep, err := NewExpect("/bin/echo", "hello world")
	if err != nil {
		t.Fatal(err)
	}
	wstr := "hello world\r\n"
	l, eerr := ep.ExpectFunc(func(a string) bool { return len(a) > 10 })
	if eerr != nil {
		t.Fatal(eerr)
	}
	if l != wstr {
		t.Fatalf(`got "%v", expected "%v"`, l, wstr)
	}
	if cerr := ep.Close(); cerr != nil {
		t.Fatal(cerr)
	}
}

func TestEcho(t *testing.T) {
	ep, err := NewExpect("/bin/echo", "hello world")
	if err != nil {
		t.Fatal(err)
	}
	l, eerr := ep.Expect("world")
	if eerr != nil {
		t.Fatal(eerr)
	}
	wstr := "hello world"
	if l[:len(wstr)] != wstr {
		t.Fatalf(`got "%v", expected "%v"`, l, wstr)
	}
	if cerr := ep.Close(); cerr != nil {
		t.Fatal(cerr)
	}
	if _, eerr = ep.Expect("..."); eerr == nil {
		t.Fatalf("expected error on closed expect process")
	}
}

func TestLineCount(t *testing.T) {
	ep, err := NewExpect("/usr/bin/printf", "1\n2\n3")
	if err != nil {
		t.Fatal(err)
	}
	wstr := "3"
	l, eerr := ep.Expect(wstr)
	if eerr != nil {
		t.Fatal(eerr)
	}
	if l != wstr {
		t.Fatalf(`got "%v", expected "%v"`, l, wstr)
	}
	if ep.LineCount() != 3 {
		t.Fatalf("got %d, expected 3", ep.LineCount())
	}
	if cerr := ep.Close(); cerr != nil {
		t.Fatal(cerr)
	}
}

func TestSend(t *testing.T) {
	ep, err := NewExpect("/usr/bin/tr", "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	if err := ep.Send("a\r"); err != nil {
		t.Fatal(err)
	}
	if _, err := ep.Expect("b"); err != nil {
		t.Fatal(err)
	}
	if err := ep.Stop(); err != nil {
		t.Fatal(err)
	}
}

func TestSignal(t *testing.T) {
	ep, err := NewExpect("/bin/sleep", "100")
	if err != nil {
		t.Fatal(err)
	}
	ep.Signal(os.Interrupt)
	donec := make(chan struct{})
	go func() {
		defer close(donec)
		werr := "signal: interrupt"
		if cerr := ep.Close(); cerr == nil || cerr.Error() != werr {
			t.Fatalf("got error %v, wanted error %s", cerr, werr)
		}
	}()
	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("signal test timed out")
	case <-donec:
	}
}
