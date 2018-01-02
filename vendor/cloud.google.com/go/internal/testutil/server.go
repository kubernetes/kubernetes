/*
Copyright 2016 Google Inc. All Rights Reserved.

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

package testutil

import (
	"net"

	grpc "google.golang.org/grpc"
)

// A Server is an in-process gRPC server, listening on a system-chosen port on
// the local loopback interface. Servers are for testing only and are not
// intended to be used in production code.
//
// To create a server, make a new Server, register your handlers, then call
// Start:
//
//	srv, err := NewServer()
//	...
//	mypb.RegisterMyServiceServer(srv.Gsrv, &myHandler)
//	....
//	srv.Start()
//
// Clients should connect to the server with no security:
//
//	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
//	...
type Server struct {
	Addr string
	l    net.Listener
	Gsrv *grpc.Server
}

// NewServer creates a new Server. The Server will be listening for gRPC connections
// at the address named by the Addr field, without TLS.
func NewServer() (*Server, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	s := &Server{
		Addr: l.Addr().String(),
		l:    l,
		Gsrv: grpc.NewServer(),
	}
	return s, nil
}

// Start causes the server to start accepting incoming connections.
// Call Start after registering handlers.
func (s *Server) Start() {
	go s.Gsrv.Serve(s.l)
}

// Close shuts down the server.
func (s *Server) Close() {
	s.Gsrv.Stop()
	s.l.Close()
}
