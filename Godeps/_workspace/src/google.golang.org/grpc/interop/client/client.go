/*
 *
 * Copyright 2014, Google Inc.
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
 */

package main

import (
	"flag"
	"io"
	"io/ioutil"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/oauth"
	"google.golang.org/grpc/grpclog"
	testpb "google.golang.org/grpc/interop/grpc_testing"
	"google.golang.org/grpc/metadata"
)

var (
	useTLS                = flag.Bool("use_tls", false, "Connection uses TLS if true, else plain TCP")
	caFile                = flag.String("tls_ca_file", "testdata/ca.pem", "The file containning the CA root cert file")
	serviceAccountKeyFile = flag.String("service_account_key_file", "", "Path to service account json key file")
	oauthScope            = flag.String("oauth_scope", "", "The scope for OAuth2 tokens")
	defaultServiceAccount = flag.String("default_service_account", "", "Email of GCE default service account")
	serverHost            = flag.String("server_host", "127.0.0.1", "The server host name")
	serverPort            = flag.Int("server_port", 10000, "The server port number")
	tlsServerName         = flag.String("server_host_override", "x.test.youtube.com", "The server name use to verify the hostname returned by TLS handshake if it is not empty. Otherwise, --server_host is used.")
	testCase              = flag.String("test_case", "large_unary",
		`Configure different test cases. Valid options are:
        empty_unary : empty (zero bytes) request and response;
        large_unary : single request and (large) response;
        client_streaming : request streaming with single response;
        server_streaming : single request with response streaming;
        ping_pong : full-duplex streaming;
        empty_stream : full-duplex streaming with zero message;
        timeout_on_sleeping_server: fullduplex streaming;
        compute_engine_creds: large_unary with compute engine auth;
        service_account_creds: large_unary with service account auth;
        jwt_token_creds: large_unary with jwt token auth;
        per_rpc_creds: large_unary with per rpc token;
        oauth2_auth_token: large_unary with oauth2 token auth;
        cancel_after_begin: cancellation after metadata has been sent but before payloads are sent;
        cancel_after_first_response: cancellation after receiving 1st message from the server.`)
)

var (
	reqSizes      = []int{27182, 8, 1828, 45904}
	respSizes     = []int{31415, 9, 2653, 58979}
	largeReqSize  = 271828
	largeRespSize = 314159
)

func newPayload(t testpb.PayloadType, size int) *testpb.Payload {
	if size < 0 {
		grpclog.Fatalf("Requested a response with invalid length %d", size)
	}
	body := make([]byte, size)
	switch t {
	case testpb.PayloadType_COMPRESSABLE:
	case testpb.PayloadType_UNCOMPRESSABLE:
		grpclog.Fatalf("PayloadType UNCOMPRESSABLE is not supported")
	default:
		grpclog.Fatalf("Unsupported payload type: %d", t)
	}
	return &testpb.Payload{
		Type: t.Enum(),
		Body: body,
	}
}

func doEmptyUnaryCall(tc testpb.TestServiceClient) {
	reply, err := tc.EmptyCall(context.Background(), &testpb.Empty{})
	if err != nil {
		grpclog.Fatal("/TestService/EmptyCall RPC failed: ", err)
	}
	if !proto.Equal(&testpb.Empty{}, reply) {
		grpclog.Fatalf("/TestService/EmptyCall receives %v, want %v", reply, testpb.Empty{})
	}
	grpclog.Println("EmptyUnaryCall done")
}

func doLargeUnaryCall(tc testpb.TestServiceClient) {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(largeRespSize)),
		Payload:      pl,
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	t := reply.GetPayload().GetType()
	s := len(reply.GetPayload().GetBody())
	if t != testpb.PayloadType_COMPRESSABLE || s != largeRespSize {
		grpclog.Fatalf("Got the reply with type %d len %d; want %d, %d", t, s, testpb.PayloadType_COMPRESSABLE, largeRespSize)
	}
	grpclog.Println("LargeUnaryCall done")
}

func doClientStreaming(tc testpb.TestServiceClient) {
	stream, err := tc.StreamingInputCall(context.Background())
	if err != nil {
		grpclog.Fatalf("%v.StreamingInputCall(_) = _, %v", tc, err)
	}
	var sum int
	for _, s := range reqSizes {
		pl := newPayload(testpb.PayloadType_COMPRESSABLE, s)
		req := &testpb.StreamingInputCallRequest{
			Payload: pl,
		}
		if err := stream.Send(req); err != nil {
			grpclog.Fatalf("%v.Send(%v) = %v", stream, req, err)
		}
		sum += s
		grpclog.Printf("Sent a request of size %d, aggregated size %d", s, sum)

	}
	reply, err := stream.CloseAndRecv()
	if err != nil {
		grpclog.Fatalf("%v.CloseAndRecv() got error %v, want %v", stream, err, nil)
	}
	if reply.GetAggregatedPayloadSize() != int32(sum) {
		grpclog.Fatalf("%v.CloseAndRecv().GetAggregatePayloadSize() = %v; want %v", stream, reply.GetAggregatedPayloadSize(), sum)
	}
	grpclog.Println("ClientStreaming done")
}

func doServerStreaming(tc testpb.TestServiceClient) {
	respParam := make([]*testpb.ResponseParameters, len(respSizes))
	for i, s := range respSizes {
		respParam[i] = &testpb.ResponseParameters{
			Size: proto.Int32(int32(s)),
		}
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
	}
	stream, err := tc.StreamingOutputCall(context.Background(), req)
	if err != nil {
		grpclog.Fatalf("%v.StreamingOutputCall(_) = _, %v", tc, err)
	}
	var rpcStatus error
	var respCnt int
	var index int
	for {
		reply, err := stream.Recv()
		if err != nil {
			rpcStatus = err
			break
		}
		t := reply.GetPayload().GetType()
		if t != testpb.PayloadType_COMPRESSABLE {
			grpclog.Fatalf("Got the reply of type %d, want %d", t, testpb.PayloadType_COMPRESSABLE)
		}
		size := len(reply.GetPayload().GetBody())
		if size != int(respSizes[index]) {
			grpclog.Fatalf("Got reply body of length %d, want %d", size, respSizes[index])
		}
		index++
		respCnt++
	}
	if rpcStatus != io.EOF {
		grpclog.Fatalf("Failed to finish the server streaming rpc: %v", err)
	}
	if respCnt != len(respSizes) {
		grpclog.Fatalf("Got %d reply, want %d", len(respSizes), respCnt)
	}
	grpclog.Println("ServerStreaming done")
}

func doPingPong(tc testpb.TestServiceClient) {
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		grpclog.Fatalf("%v.FullDuplexCall(_) = _, %v", tc, err)
	}
	var index int
	for index < len(reqSizes) {
		respParam := []*testpb.ResponseParameters{
			{
				Size: proto.Int32(int32(respSizes[index])),
			},
		}
		pl := newPayload(testpb.PayloadType_COMPRESSABLE, reqSizes[index])
		req := &testpb.StreamingOutputCallRequest{
			ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
			ResponseParameters: respParam,
			Payload:            pl,
		}
		if err := stream.Send(req); err != nil {
			grpclog.Fatalf("%v.Send(%v) = %v", stream, req, err)
		}
		reply, err := stream.Recv()
		if err != nil {
			grpclog.Fatalf("%v.Recv() = %v", stream, err)
		}
		t := reply.GetPayload().GetType()
		if t != testpb.PayloadType_COMPRESSABLE {
			grpclog.Fatalf("Got the reply of type %d, want %d", t, testpb.PayloadType_COMPRESSABLE)
		}
		size := len(reply.GetPayload().GetBody())
		if size != int(respSizes[index]) {
			grpclog.Fatalf("Got reply body of length %d, want %d", size, respSizes[index])
		}
		index++
	}
	if err := stream.CloseSend(); err != nil {
		grpclog.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		grpclog.Fatalf("%v failed to complele the ping pong test: %v", stream, err)
	}
	grpclog.Println("Pingpong done")
}

func doEmptyStream(tc testpb.TestServiceClient) {
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		grpclog.Fatalf("%v.FullDuplexCall(_) = _, %v", tc, err)
	}
	if err := stream.CloseSend(); err != nil {
		grpclog.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		grpclog.Fatalf("%v failed to complete the empty stream test: %v", stream, err)
	}
	grpclog.Println("Emptystream done")
}

func doTimeoutOnSleepingServer(tc testpb.TestServiceClient) {
	ctx, _ := context.WithTimeout(context.Background(), 1*time.Millisecond)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		if grpc.Code(err) == codes.DeadlineExceeded {
			grpclog.Println("TimeoutOnSleepingServer done")
			return
		}
		grpclog.Fatalf("%v.FullDuplexCall(_) = _, %v", tc, err)
	}
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, 27182)
	req := &testpb.StreamingOutputCallRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		Payload:      pl,
	}
	if err := stream.Send(req); err != nil {
		grpclog.Fatalf("%v.Send(%v) = %v", stream, req, err)
	}
	if _, err := stream.Recv(); grpc.Code(err) != codes.DeadlineExceeded {
		grpclog.Fatalf("%v.Recv() = _, %v, want error code %d", stream, err, codes.DeadlineExceeded)
	}
	grpclog.Println("TimeoutOnSleepingServer done")
}

func doComputeEngineCreds(tc testpb.TestServiceClient) {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize:   proto.Int32(int32(largeRespSize)),
		Payload:        pl,
		FillUsername:   proto.Bool(true),
		FillOauthScope: proto.Bool(true),
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if user != *defaultServiceAccount {
		grpclog.Fatalf("Got user name %q, want %q.", user, *defaultServiceAccount)
	}
	if !strings.Contains(*oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, *oauthScope)
	}
	grpclog.Println("ComputeEngineCreds done")
}

func getServiceAccountJSONKey() []byte {
	jsonKey, err := ioutil.ReadFile(*serviceAccountKeyFile)
	if err != nil {
		grpclog.Fatalf("Failed to read the service account key file: %v", err)
	}
	return jsonKey
}

func doServiceAccountCreds(tc testpb.TestServiceClient) {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize:   proto.Int32(int32(largeRespSize)),
		Payload:        pl,
		FillUsername:   proto.Bool(true),
		FillOauthScope: proto.Bool(true),
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	jsonKey := getServiceAccountJSONKey()
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	if !strings.Contains(*oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, *oauthScope)
	}
	grpclog.Println("ServiceAccountCreds done")
}

func doJWTTokenCreds(tc testpb.TestServiceClient) {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(largeRespSize)),
		Payload:      pl,
		FillUsername: proto.Bool(true),
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	jsonKey := getServiceAccountJSONKey()
	user := reply.GetUsername()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	grpclog.Println("JWTtokenCreds done")
}

func getToken() *oauth2.Token {
	jsonKey := getServiceAccountJSONKey()
	config, err := google.JWTConfigFromJSON(jsonKey, *oauthScope)
	if err != nil {
		grpclog.Fatalf("Failed to get the config: %v", err)
	}
	token, err := config.TokenSource(context.Background()).Token()
	if err != nil {
		grpclog.Fatalf("Failed to get the token: %v", err)
	}
	return token
}

func doOauth2TokenCreds(tc testpb.TestServiceClient) {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize:   proto.Int32(int32(largeRespSize)),
		Payload:        pl,
		FillUsername:   proto.Bool(true),
		FillOauthScope: proto.Bool(true),
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	jsonKey := getServiceAccountJSONKey()
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	if !strings.Contains(*oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, *oauthScope)
	}
	grpclog.Println("Oauth2TokenCreds done")
}

func doPerRPCCreds(tc testpb.TestServiceClient) {
	jsonKey := getServiceAccountJSONKey()
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize:   proto.Int32(int32(largeRespSize)),
		Payload:        pl,
		FillUsername:   proto.Bool(true),
		FillOauthScope: proto.Bool(true),
	}
	token := getToken()
	kv := map[string]string{"authorization": token.TokenType + " " + token.AccessToken}
	ctx := metadata.NewContext(context.Background(), metadata.MD{"authorization": []string{kv["authorization"]}})
	reply, err := tc.UnaryCall(ctx, req)
	if err != nil {
		grpclog.Fatal("/TestService/UnaryCall RPC failed: ", err)
	}
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	if !strings.Contains(*oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, *oauthScope)
	}
	grpclog.Println("PerRPCCreds done")
}

var (
	testMetadata = metadata.MD{
		"key1": []string{"value1"},
		"key2": []string{"value2"},
	}
)

func doCancelAfterBegin(tc testpb.TestServiceClient) {
	ctx, cancel := context.WithCancel(metadata.NewContext(context.Background(), testMetadata))
	stream, err := tc.StreamingInputCall(ctx)
	if err != nil {
		grpclog.Fatalf("%v.StreamingInputCall(_) = _, %v", tc, err)
	}
	cancel()
	_, err = stream.CloseAndRecv()
	if grpc.Code(err) != codes.Canceled {
		grpclog.Fatalf("%v.CloseAndRecv() got error code %d, want %d", stream, grpc.Code(err), codes.Canceled)
	}
	grpclog.Println("CancelAfterBegin done")
}

func doCancelAfterFirstResponse(tc testpb.TestServiceClient) {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		grpclog.Fatalf("%v.FullDuplexCall(_) = _, %v", tc, err)
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(31415),
		},
	}
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, 27182)
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            pl,
	}
	if err := stream.Send(req); err != nil {
		grpclog.Fatalf("%v.Send(%v) = %v", stream, req, err)
	}
	if _, err := stream.Recv(); err != nil {
		grpclog.Fatalf("%v.Recv() = %v", stream, err)
	}
	cancel()
	if _, err := stream.Recv(); grpc.Code(err) != codes.Canceled {
		grpclog.Fatalf("%v compleled with error code %d, want %d", stream, grpc.Code(err), codes.Canceled)
	}
	grpclog.Println("CancelAfterFirstResponse done")
}

func main() {
	flag.Parse()
	serverAddr := net.JoinHostPort(*serverHost, strconv.Itoa(*serverPort))
	var opts []grpc.DialOption
	if *useTLS {
		var sn string
		if *tlsServerName != "" {
			sn = *tlsServerName
		}
		var creds credentials.TransportAuthenticator
		if *caFile != "" {
			var err error
			creds, err = credentials.NewClientTLSFromFile(*caFile, sn)
			if err != nil {
				grpclog.Fatalf("Failed to create TLS credentials %v", err)
			}
		} else {
			creds = credentials.NewClientTLSFromCert(nil, sn)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
		if *testCase == "compute_engine_creds" {
			opts = append(opts, grpc.WithPerRPCCredentials(oauth.NewComputeEngine()))
		} else if *testCase == "service_account_creds" {
			jwtCreds, err := oauth.NewServiceAccountFromFile(*serviceAccountKeyFile, *oauthScope)
			if err != nil {
				grpclog.Fatalf("Failed to create JWT credentials: %v", err)
			}
			opts = append(opts, grpc.WithPerRPCCredentials(jwtCreds))
		} else if *testCase == "jwt_token_creds" {
			jwtCreds, err := oauth.NewJWTAccessFromFile(*serviceAccountKeyFile)
			if err != nil {
				grpclog.Fatalf("Failed to create JWT credentials: %v", err)
			}
			opts = append(opts, grpc.WithPerRPCCredentials(jwtCreds))
		} else if *testCase == "oauth2_auth_token" {
			opts = append(opts, grpc.WithPerRPCCredentials(oauth.NewOauthAccess(getToken())))
		}
	} else {
		opts = append(opts, grpc.WithInsecure())
	}
	conn, err := grpc.Dial(serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("Fail to dial: %v", err)
	}
	defer conn.Close()
	tc := testpb.NewTestServiceClient(conn)
	switch *testCase {
	case "empty_unary":
		doEmptyUnaryCall(tc)
	case "large_unary":
		doLargeUnaryCall(tc)
	case "client_streaming":
		doClientStreaming(tc)
	case "server_streaming":
		doServerStreaming(tc)
	case "ping_pong":
		doPingPong(tc)
	case "empty_stream":
		doEmptyStream(tc)
	case "timeout_on_sleeping_server":
		doTimeoutOnSleepingServer(tc)
	case "compute_engine_creds":
		if !*useTLS {
			grpclog.Fatalf("TLS is not enabled. TLS is required to execute compute_engine_creds test case.")
		}
		doComputeEngineCreds(tc)
	case "service_account_creds":
		if !*useTLS {
			grpclog.Fatalf("TLS is not enabled. TLS is required to execute service_account_creds test case.")
		}
		doServiceAccountCreds(tc)
	case "jwt_token_creds":
		if !*useTLS {
			grpclog.Fatalf("TLS is not enabled. TLS is required to execute jwt_token_creds test case.")
		}
		doJWTTokenCreds(tc)
	case "per_rpc_creds":
		if !*useTLS {
			grpclog.Fatalf("TLS is not enabled. TLS is required to execute per_rpc_creds test case.")
		}
		doPerRPCCreds(tc)
	case "oauth2_auth_token":
		if !*useTLS {
			grpclog.Fatalf("TLS is not enabled. TLS is required to execute oauth2_auth_token test case.")
		}
		doOauth2TokenCreds(tc)
	case "cancel_after_begin":
		doCancelAfterBegin(tc)
	case "cancel_after_first_response":
		doCancelAfterFirstResponse(tc)
	default:
		grpclog.Fatal("Unsupported test case: ", *testCase)
	}
}
