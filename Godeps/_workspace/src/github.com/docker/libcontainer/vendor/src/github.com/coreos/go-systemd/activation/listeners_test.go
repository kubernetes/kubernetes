/*
Copyright 2014 CoreOS Inc.

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

package activation

import (
	"io"
	"net"
	"os"
	"os/exec"
	"testing"
)

// correctStringWritten fails the text if the correct string wasn't written
// to the other side of the pipe.
func correctStringWrittenNet(t *testing.T, r net.Conn, expected string) bool {
	bytes := make([]byte, len(expected))
	io.ReadAtLeast(r, bytes, len(expected))

	if string(bytes) != expected {
		t.Fatalf("Unexpected string %s", string(bytes))
	}

	return true
}

// TestActivation forks out a copy of activation.go example and reads back two
// strings from the pipes that are passed in.
func TestListeners(t *testing.T) {
	cmd := exec.Command("go", "run", "../examples/activation/listen.go")

	l1, err := net.Listen("tcp", ":9999")
	if err != nil {
		t.Fatalf(err.Error())
	}
	l2, err := net.Listen("tcp", ":1234")
	if err != nil {
		t.Fatalf(err.Error())
	}

	t1 := l1.(*net.TCPListener)
	t2 := l2.(*net.TCPListener)

	f1, _ := t1.File()
	f2, _  := t2.File()

	cmd.ExtraFiles = []*os.File{
		f1,
		f2,
	}

	r1, err := net.Dial("tcp", "127.0.0.1:9999")
	if err != nil {
		t.Fatalf(err.Error())
	}
	r1.Write([]byte("Hi"))

	r2, err := net.Dial("tcp", "127.0.0.1:1234")
	if err != nil {
		t.Fatalf(err.Error())
	}
	r2.Write([]byte("Hi"))

	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LISTEN_FDS=2", "FIX_LISTEN_PID=1")

	out, err := cmd.Output()
	if err != nil {
		println(string(out))
		t.Fatalf(err.Error())
	}

	correctStringWrittenNet(t, r1, "Hello world")
	correctStringWrittenNet(t, r2, "Goodbye world")
}
