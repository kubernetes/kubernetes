// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grpc

import (
	"context"
	"net"
	"testing"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

func TestPool(t *testing.T) {
	conn1 := &grpc.ClientConn{}
	conn2 := &grpc.ClientConn{}

	pool := &roundRobinConnPool{
		conns: []*grpc.ClientConn{
			conn1, conn2,
		},
	}

	if got := pool.Conn(); got != conn2 {
		t.Errorf("pool.Conn() #1 got %v; want conn2 (%v)", got, conn1)
	}

	if got := pool.Conn(); got != conn1 {
		t.Errorf("pool.Conn() #2 got %v; want conn1 (%v)", got, conn1)
	}

	if got := pool.Conn(); got != conn2 {
		t.Errorf("pool.Conn() #3 got %v; want conn2 (%v)", got, conn1)
	}
}

func TestClose(t *testing.T) {
	_, l := mockServer(t)

	pool := &roundRobinConnPool{}
	for i := 0; i < 4; i++ {
		conn, err := grpc.Dial(l.Addr().String(), grpc.WithInsecure())
		if err != nil {
			t.Fatal(err)
		}
		pool.conns = append(pool.conns, conn)
	}

	if err := pool.Close(); err != nil {
		t.Fatalf("pool.Close: %v", err)
	}
}

// Regression test for https://github.com/googleapis/google-cloud-go/issues/1780
func TestWithGRPCConnAndPoolSize(t *testing.T) {
	_, l := mockServer(t)

	conn, err := grpc.Dial(l.Addr().String(), grpc.WithInsecure())
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	connPool, err := DialPool(ctx,
		option.WithGRPCConn(conn),
		option.WithGRPCConnectionPool(4),
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := connPool.Close(); err != nil {
		t.Fatalf("pool.Close: %v", err)
	}
}

func mockServer(t *testing.T) (*grpc.Server, net.Listener) {
	t.Helper()

	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}

	s := grpc.NewServer()
	go s.Serve(l)

	return s, l
}
