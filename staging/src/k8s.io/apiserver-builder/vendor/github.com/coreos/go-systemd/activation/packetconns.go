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
)

// PacketConns returns a slice containing a net.PacketConn for each matching socket type
// passed to this process.
//
// The order of the file descriptors is preserved in the returned slice.
// Nil values are used to fill any gaps. For example if systemd were to return file descriptors
// corresponding with "udp, tcp, udp", then the slice would contain {net.PacketConn, nil, net.PacketConn}
func PacketConns(unsetEnv bool) ([]net.PacketConn, error) {
	files := Files(unsetEnv)
	conns := make([]net.PacketConn, len(files))

	for i, f := range files {
		if pc, err := net.FilePacketConn(f); err == nil {
			conns[i] = pc
		}
	}
	return conns, nil
}
