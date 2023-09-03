//go:build !windows
// +build !windows

/*
Copyright 2023 The Kubernetes Authors.

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

package probe

import (
	"net"
	"syscall"
)

// ProbeDialer returns a dialer optimized for probes to avoid lingering sockets on TIME-WAIT state.
// The dialer reduces the TIME-WAIT period to 1 seconds instead of the OS default of 60 seconds.
// Using 1 second instead of 0 because SO_LINGER socket option to 0 causes pending data to be
// discarded and the connection to be aborted with an RST rather than for the pending data to be
// transmitted and the connection closed cleanly with a FIN.
// Ref: https://issues.k8s.io/89898
func ProbeDialer() *net.Dialer {
	dialer := &net.Dialer{
		Control: func(network, address string, c syscall.RawConn) error {
			return c.Control(func(fd uintptr) {
				syscall.SetsockoptLinger(int(fd), syscall.SOL_SOCKET, syscall.SO_LINGER, &syscall.Linger{Onoff: 1, Linger: 1})
			})
		},
	}
	return dialer
}
