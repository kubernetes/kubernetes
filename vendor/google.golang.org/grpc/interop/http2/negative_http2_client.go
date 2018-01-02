/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * Client used to test http2 error edge cases like GOAWAYs and RST_STREAMs
 *
 * Documentation:
 *	 https://github.com/grpc/grpc/blob/master/doc/negative-http2-interop-test-descriptions.md
 */

package main

import (
	"flag"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/interop"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

var (
	serverHost = flag.String("server_host", "127.0.0.1", "The server host name")
	serverPort = flag.Int("server_port", 8080, "The server port number")
	testCase   = flag.String("test_case", "goaway",
		`Configure different test cases. Valid options are:
        goaway : client sends two requests, the server will send a goaway in between;
        rst_after_header : server will send rst_stream after it sends headers;
        rst_during_data : server will send rst_stream while sending data;
        rst_after_data : server will send rst_stream after sending data;
        ping : server will send pings between each http2 frame;
        max_streams : server will ensure that the max_concurrent_streams limit is upheld;`)
	largeReqSize  = 271828
	largeRespSize = 314159
)

func largeSimpleRequest() *testpb.SimpleRequest {
	pl := interop.ClientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	return &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(largeRespSize)),
		Payload:      pl,
	}
}

// sends two unary calls. The server asserts that the calls use different connections.
func goaway(tc testpb.TestServiceClient) {
	interop.DoLargeUnaryCall(tc)
	// sleep to ensure that the client has time to recv the GOAWAY.
	// TODO(ncteisen): make this less hacky.
	time.Sleep(1 * time.Second)
	interop.DoLargeUnaryCall(tc)
}

func rstAfterHeader(tc testpb.TestServiceClient) {
	req := largeSimpleRequest()
	reply, err := tc.UnaryCall(context.Background(), req)
	if reply != nil {
		grpclog.Fatalf("Client received reply despite server sending rst stream after header")
	}
	if grpc.Code(err) != codes.Internal {
		grpclog.Fatalf("%v.UnaryCall() = _, %v, want _, %v", tc, grpc.Code(err), codes.Internal)
	}
}

func rstDuringData(tc testpb.TestServiceClient) {
	req := largeSimpleRequest()
	reply, err := tc.UnaryCall(context.Background(), req)
	if reply != nil {
		grpclog.Fatalf("Client received reply despite server sending rst stream during data")
	}
	if grpc.Code(err) != codes.Unknown {
		grpclog.Fatalf("%v.UnaryCall() = _, %v, want _, %v", tc, grpc.Code(err), codes.Unknown)
	}
}

func rstAfterData(tc testpb.TestServiceClient) {
	req := largeSimpleRequest()
	reply, err := tc.UnaryCall(context.Background(), req)
	if reply != nil {
		grpclog.Fatalf("Client received reply despite server sending rst stream after data")
	}
	if grpc.Code(err) != codes.Internal {
		grpclog.Fatalf("%v.UnaryCall() = _, %v, want _, %v", tc, grpc.Code(err), codes.Internal)
	}
}

func ping(tc testpb.TestServiceClient) {
	// The server will assert that every ping it sends was ACK-ed by the client.
	interop.DoLargeUnaryCall(tc)
}

func maxStreams(tc testpb.TestServiceClient) {
	interop.DoLargeUnaryCall(tc)
	var wg sync.WaitGroup
	for i := 0; i < 15; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			interop.DoLargeUnaryCall(tc)
		}()
	}
	wg.Wait()
}

func main() {
	flag.Parse()
	serverAddr := net.JoinHostPort(*serverHost, strconv.Itoa(*serverPort))
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	conn, err := grpc.Dial(serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("Fail to dial: %v", err)
	}
	defer conn.Close()
	tc := testpb.NewTestServiceClient(conn)
	switch *testCase {
	case "goaway":
		goaway(tc)
		grpclog.Println("goaway done")
	case "rst_after_header":
		rstAfterHeader(tc)
		grpclog.Println("rst_after_header done")
	case "rst_during_data":
		rstDuringData(tc)
		grpclog.Println("rst_during_data done")
	case "rst_after_data":
		rstAfterData(tc)
		grpclog.Println("rst_after_data done")
	case "ping":
		ping(tc)
		grpclog.Println("ping done")
	case "max_streams":
		maxStreams(tc)
		grpclog.Println("max_streams done")
	default:
		grpclog.Fatal("Unsupported test case: ", *testCase)
	}
}
