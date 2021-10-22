/*
 *
 * Copyright 2014 gRPC authors.
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

package main

import (
	"flag"
	"net"
	"strconv"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/alts"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/interop"
	testpb "google.golang.org/grpc/interop/grpc_testing"
	"google.golang.org/grpc/testdata"
)

var (
	useTLS     = flag.Bool("use_tls", false, "Connection uses TLS if true, else plain TCP")
	useALTS    = flag.Bool("use_alts", false, "Connection uses ALTS if true (this option can only be used on GCP)")
	altsHSAddr = flag.String("alts_handshaker_service_address", "", "ALTS handshaker gRPC service address")
	certFile   = flag.String("tls_cert_file", "", "The TLS cert file")
	keyFile    = flag.String("tls_key_file", "", "The TLS key file")
	port       = flag.Int("port", 10000, "The server port")
)

func main() {
	flag.Parse()
	if *useTLS && *useALTS {
		grpclog.Fatalf("use_tls and use_alts cannot be both set to true")
	}
	p := strconv.Itoa(*port)
	lis, err := net.Listen("tcp", ":"+p)
	if err != nil {
		grpclog.Fatalf("failed to listen: %v", err)
	}
	var opts []grpc.ServerOption
	if *useTLS {
		if *certFile == "" {
			*certFile = testdata.Path("server1.pem")
		}
		if *keyFile == "" {
			*keyFile = testdata.Path("server1.key")
		}
		creds, err := credentials.NewServerTLSFromFile(*certFile, *keyFile)
		if err != nil {
			grpclog.Fatalf("Failed to generate credentials %v", err)
		}
		opts = append(opts, grpc.Creds(creds))
	} else if *useALTS {
		altsOpts := alts.DefaultServerOptions()
		if *altsHSAddr != "" {
			altsOpts.HandshakerServiceAddress = *altsHSAddr
		}
		altsTC := alts.NewServerCreds(altsOpts)
		opts = append(opts, grpc.Creds(altsTC))
	}
	server := grpc.NewServer(opts...)
	testpb.RegisterTestServiceServer(server, interop.NewTestServer())
	server.Serve(lis)
}
