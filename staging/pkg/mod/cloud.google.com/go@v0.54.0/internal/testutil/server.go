/*
Copyright 2016 Google LLC

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
	"fmt"
	"log"
	"net"
	"regexp"
	"strconv"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
	Port int
	l    net.Listener
	Gsrv *grpc.Server
}

// NewServer creates a new Server. The Server will be listening for gRPC connections
// at the address named by the Addr field, without TLS.
func NewServer(opts ...grpc.ServerOption) (*Server, error) {
	return NewServerWithPort(0, opts...)
}

// NewServerWithPort creates a new Server at a specific port. The Server will be listening
// for gRPC connections at the address named by the Addr field, without TLS.
func NewServerWithPort(port int, opts ...grpc.ServerOption) (*Server, error) {
	l, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", port))
	if err != nil {
		return nil, err
	}
	s := &Server{
		Addr: l.Addr().String(),
		Port: parsePort(l.Addr().String()),
		l:    l,
		Gsrv: grpc.NewServer(opts...),
	}
	return s, nil
}

// Start causes the server to start accepting incoming connections.
// Call Start after registering handlers.
func (s *Server) Start() {
	go func() {
		if err := s.Gsrv.Serve(s.l); err != nil {
			log.Printf("testutil.Server.Start: %v", err)
		}
	}()
}

// Close shuts down the server.
func (s *Server) Close() {
	s.Gsrv.Stop()
	s.l.Close()
}

// PageBounds converts an incoming page size and token from an RPC request into
// slice bounds and the outgoing next-page token.
//
// PageBounds assumes that the complete, unpaginated list of items exists as a
// single slice. In addition to the page size and token, PageBounds needs the
// length of that slice.
//
// PageBounds's first two return values should be used to construct a sub-slice of
// the complete, unpaginated slice. E.g. if the complete slice is s, then
// s[from:to] is the desired page. Its third return value should be set as the
// NextPageToken field of the RPC response.
func PageBounds(pageSize int, pageToken string, length int) (from, to int, nextPageToken string, err error) {
	from, to = 0, length
	if pageToken != "" {
		from, err = strconv.Atoi(pageToken)
		if err != nil {
			return 0, 0, "", status.Errorf(codes.InvalidArgument, "bad page token: %v", err)
		}
		if from >= length {
			return length, length, "", nil
		}
	}
	if pageSize > 0 && from+pageSize < length {
		to = from + pageSize
		nextPageToken = strconv.Itoa(to)
	}
	return from, to, nextPageToken, nil
}

var portParser = regexp.MustCompile(`:[0-9]+`)

func parsePort(addr string) int {
	res := portParser.FindAllString(addr, -1)
	if len(res) == 0 {
		panic(fmt.Errorf("parsePort: found no numbers in %s", addr))
	}
	stringPort := res[0][1:] // strip the :
	p, err := strconv.ParseInt(stringPort, 10, 32)
	if err != nil {
		panic(err)
	}
	return int(p)
}
