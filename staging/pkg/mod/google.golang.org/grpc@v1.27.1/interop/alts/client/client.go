/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// This binary can only run on Google Cloud Platform (GCP).
package main

import (
	"context"
	"flag"
	"time"

	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/alts"
	"google.golang.org/grpc/grpclog"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

var (
	hsAddr     = flag.String("alts_handshaker_service_address", "", "ALTS handshaker gRPC service address")
	serverAddr = flag.String("server_address", ":8080", "The port on which the server is listening")
)

func main() {
	flag.Parse()

	opts := alts.DefaultClientOptions()
	if *hsAddr != "" {
		opts.HandshakerServiceAddress = *hsAddr
	}
	altsTC := alts.NewClientCreds(opts)
	// Block until the server is ready.
	conn, err := grpc.Dial(*serverAddr, grpc.WithTransportCredentials(altsTC), grpc.WithBlock())
	if err != nil {
		grpclog.Fatalf("gRPC Client: failed to dial the server at %v: %v", *serverAddr, err)
	}
	defer conn.Close()
	grpcClient := testpb.NewTestServiceClient(conn)

	// Call the EmptyCall API.
	ctx := context.Background()
	request := &testpb.Empty{}
	if _, err := grpcClient.EmptyCall(ctx, request); err != nil {
		grpclog.Fatalf("grpc Client: EmptyCall(_, %v) failed: %v", request, err)
	}
	grpclog.Info("grpc Client: empty call succeeded")

	// This sleep prevents the connection from being abruptly disconnected
	// when running this binary (along with grpc_server) on GCP dev cluster.
	time.Sleep(1 * time.Second)
}
