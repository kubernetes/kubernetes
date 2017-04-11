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

package activation

import (
	"net"
	"os"
	"os/exec"
	"testing"
)

// TestActivation forks out a copy of activation.go example and reads back two
// strings from the pipes that are passed in.
func TestPacketConns(t *testing.T) {
	cmd := exec.Command("go", "run", "../examples/activation/udpconn.go")

	u1, err := net.ListenUDP("udp", &net.UDPAddr{Port: 9999})
	if err != nil {
		t.Fatalf(err.Error())
	}
	u2, err := net.ListenUDP("udp", &net.UDPAddr{Port: 1234})
	if err != nil {
		t.Fatalf(err.Error())
	}

	f1, _ := u1.File()
	f2, _ := u2.File()

	cmd.ExtraFiles = []*os.File{
		f1,
		f2,
	}

	r1, err := net.Dial("udp", "127.0.0.1:9999")
	if err != nil {
		t.Fatalf(err.Error())
	}
	r1.Write([]byte("Hi"))

	r2, err := net.Dial("udp", "127.0.0.1:1234")
	if err != nil {
		t.Fatalf(err.Error())
	}
	r2.Write([]byte("Hi"))

	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LISTEN_FDS=2", "FIX_LISTEN_PID=1")

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd output '%s', err: '%s'\n", out, err)
	}

	correctStringWrittenNet(t, r1, "Hello world")
	correctStringWrittenNet(t, r2, "Goodbye world")
}
