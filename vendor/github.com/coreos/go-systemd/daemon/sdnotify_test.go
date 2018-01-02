// Copyright 2016 CoreOS, Inc.
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

package daemon

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"testing"
)

// TestSdNotify
func TestSdNotify(t *testing.T) {

	testDir, e := ioutil.TempDir("/tmp/", "test-")
	if e != nil {
		panic(e)
	}
	defer os.RemoveAll(testDir)

	notifySocket := testDir + "/notify-socket.sock"
	laddr := net.UnixAddr{
		Name: notifySocket,
		Net:  "unixgram",
	}
	_, e = net.ListenUnixgram("unixgram", &laddr)
	if e != nil {
		panic(e)
	}

	tests := []struct {
		unsetEnv  bool
		envSocket string

		wsent bool
		werr  bool
	}{
		// (true, nil) - notification supported, data has been sent
		{false, notifySocket, true, false},
		// (false, err) - notification supported, but failure happened
		{true, testDir + "/missing.sock", false, true},
		// (false, nil) - notification not supported
		{true, "", false, false},
	}

	for i, tt := range tests {
		must(os.Unsetenv("NOTIFY_SOCKET"))
		if tt.envSocket != "" {
			must(os.Setenv("NOTIFY_SOCKET", tt.envSocket))
		}
		sent, err := SdNotify(tt.unsetEnv, fmt.Sprintf("TestSdNotify test message #%d", i))

		if sent != tt.wsent {
			t.Errorf("#%d: expected send result %t, got %t", i, tt.wsent, sent)
		}
		if tt.werr && err == nil {
			t.Errorf("#%d: want non-nil err, got nil", i)
		} else if !tt.werr && err != nil {
			t.Errorf("#%d: want nil err, got %v", i, err)
		}
		if tt.unsetEnv && tt.envSocket != "" && os.Getenv("NOTIFY_SOCKET") != "" {
			t.Errorf("#%d: environment variable not cleaned up", i)
		}

	}
}
