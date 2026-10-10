//go:build !windows

/*
Copyright The Kubernetes Authors.

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

package remotecommand

import (
	"net"
	"syscall"
)

// smallReceiveBufferDialer returns a dialer whose sockets have a small, fixed
// receive buffer. Output the client has not read yet then backs up in the
// server's send buffer rather than in the client's receive buffer, which is
// what a server that closes the socket too early discards; without it the
// kernel's receive buffer autotuning can absorb the whole output and hide the
// loss.
func smallReceiveBufferDialer() *net.Dialer {
	return &net.Dialer{
		Control: func(_, _ string, c syscall.RawConn) error {
			var err error
			if cerr := c.Control(func(fd uintptr) {
				err = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_RCVBUF, 64<<10)
			}); cerr != nil {
				return cerr
			}
			return err
		},
	}
}
