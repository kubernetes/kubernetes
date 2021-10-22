// Copyright 2018, OpenCensus Authors
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

package testpb

import (
	"context"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	"go.opencensus.io/plugin/ocgrpc"
	"go.opencensus.io/trace"
	"google.golang.org/grpc"
)

type testServer struct{}

var _ FooServer = (*testServer)(nil)

func (s *testServer) Single(ctx context.Context, in *FooRequest) (*FooResponse, error) {
	if in.SleepNanos > 0 {
		_, span := trace.StartSpan(ctx, "testpb.Single.Sleep")
		span.AddAttributes(trace.Int64Attribute("sleep_nanos", in.SleepNanos))
		time.Sleep(time.Duration(in.SleepNanos))
		span.End()
	}
	if in.Fail {
		return nil, fmt.Errorf("request failed")
	}
	return &FooResponse{}, nil
}

func (s *testServer) Multiple(stream Foo_MultipleServer) error {
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if in.Fail {
			return fmt.Errorf("request failed")
		}
		if err := stream.Send(&FooResponse{}); err != nil {
			return err
		}
	}
}

// NewTestClient returns a new TestClient.
func NewTestClient(l *testing.T) (client FooClient, cleanup func()) {
	// initialize server
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		l.Fatal(err)
	}
	server := grpc.NewServer(grpc.StatsHandler(&ocgrpc.ServerHandler{}))
	RegisterFooServer(server, &testServer{})
	go server.Serve(listener)

	// Initialize client.
	clientConn, err := grpc.Dial(
		listener.Addr().String(),
		grpc.WithInsecure(),
		grpc.WithStatsHandler(&ocgrpc.ClientHandler{}),
		grpc.WithBlock())

	if err != nil {
		l.Fatal(err)
	}
	client = NewFooClient(clientConn)

	cleanup = func() {
		server.GracefulStop()
		clientConn.Close()
	}

	return client, cleanup
}
