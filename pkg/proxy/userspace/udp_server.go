/*
Copyright 2014 The Kubernetes Authors.

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

package userspace

import (
	"fmt"
	"net"
)

// udpEchoServer is a simple echo server in UDP, intended for testing the proxy.
type udpEchoServer struct {
	net.PacketConn
}

func (r *udpEchoServer) Loop() {
	var buffer [4096]byte
	for {
		n, cliAddr, err := r.ReadFrom(buffer[0:])
		if err != nil {
			fmt.Printf("ReadFrom failed: %v\n", err)
			continue
		}
		r.WriteTo(buffer[0:n], cliAddr)
	}
}

func newUDPEchoServer() (*udpEchoServer, error) {
	packetconn, err := net.ListenPacket("udp", ":0")
	if err != nil {
		return nil, err
	}
	return &udpEchoServer{packetconn}, nil
}

/*
func main() {
	r,_ := newUDPEchoServer()
	r.Loop()
}
*/
