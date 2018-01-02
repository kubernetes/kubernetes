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
	"crypto/tls"
	"net"
)

// Listeners returns a slice containing a net.Listener for each matching socket type
// passed to this process.
//
// The order of the file descriptors is preserved in the returned slice.
// Nil values are used to fill any gaps. For example if systemd were to return file descriptors
// corresponding with "udp, tcp, tcp", then the slice would contain {nil, net.Listener, net.Listener}
func Listeners(unsetEnv bool) ([]net.Listener, error) {
	files := Files(unsetEnv)
	listeners := make([]net.Listener, len(files))

	for i, f := range files {
		if pc, err := net.FileListener(f); err == nil {
			listeners[i] = pc
		}
	}
	return listeners, nil
}

// TLSListeners returns a slice containing a net.listener for each matching TCP socket type
// passed to this process.
// It uses default Listeners func and forces TCP sockets handlers to use TLS based on tlsConfig.
func TLSListeners(unsetEnv bool, tlsConfig *tls.Config) ([]net.Listener, error) {
	listeners, err := Listeners(unsetEnv)

	if listeners == nil || err != nil {
		return nil, err
	}

	if tlsConfig != nil && err == nil {
		for i, l := range listeners {
			// Activate TLS only for TCP sockets
			if l.Addr().Network() == "tcp" {
				listeners[i] = tls.NewListener(l, tlsConfig)
			}
		}
	}

	return listeners, err
}
