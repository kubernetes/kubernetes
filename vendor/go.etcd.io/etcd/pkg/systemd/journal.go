// Copyright 2018 The etcd Authors
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

package systemd

import "net"

// DialJournal returns no error if the process can dial journal socket.
// Returns an error if dial failed, whichi indicates journald is not available
// (e.g. run embedded etcd as docker daemon).
// Reference: https://github.com/coreos/go-systemd/blob/master/journal/journal.go.
func DialJournal() error {
	conn, err := net.Dial("unixgram", "/run/systemd/journal/socket")
	if conn != nil {
		defer conn.Close()
	}
	return err
}
