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

package interop

import (
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	testpb "google.golang.org/grpc/interop/grpc_testing"
	"google.golang.org/grpc/metadata"
)

var (
	reqSizes      = []int{27182, 8, 1828, 45904}
	respSizes     = []int{31415, 9, 2653, 58979}
	largeReqSize  = 271828
	largeRespSize = 314159
)

func clientNewPayload(t testpb.PayloadType, size int) *testpb.Payload {
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

// DoEmptyUnaryCall performs a unary RPC with empty request and response messages.
func DoEmptyUnaryCall(tc testpb.TestServiceClient) {
	reply, err := tc.EmptyCall(context.Background(), &testpb.Empty{})
	if err != nil {
		grpclog.Fatal("/TestService/EmptyCall RPC failed: ", err)
	}
	if !proto.Equal(&testpb.Empty{}, reply) {
		grpclog.Fatalf("/TestService/EmptyCall receives %v, want %v", reply, testpb.Empty{})
	}
}

// DoLargeUnaryCall performs a unary RPC with large payload in the request and response.
func DoLargeUnaryCall(tc testpb.TestServiceClient) {
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
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
}

// DoClientStreaming performs a client streaming RPC.
func DoClientStreaming(tc testpb.TestServiceClient) {
	stream, err := tc.StreamingInputCall(context.Background())
	if err != nil {
		grpclog.Fatalf("%v.StreamingInputCall(_) = _, %v", tc, err)
	}
	var sum int
	for _, s := range reqSizes {
		pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, s)
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
}

// DoServerStreaming performs a server streaming RPC.
func DoServerStreaming(tc testpb.TestServiceClient) {
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
}

// DoPingPong performs ping-pong style bi-directional streaming RPC.
func DoPingPong(tc testpb.TestServiceClient) {
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
		pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, reqSizes[index])
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
}

// DoEmptyStream sets up a bi-directional streaming with zero message.
func DoEmptyStream(tc testpb.TestServiceClient) {
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
}

// DoTimeoutOnSleepingServer performs an RPC on a sleep server which causes RPC timeout.
func DoTimeoutOnSleepingServer(tc testpb.TestServiceClient) {
	ctx, _ := context.WithTimeout(context.Background(), 1*time.Millisecond)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		if grpc.Code(err) == codes.DeadlineExceeded {
			return
		}
		grpclog.Fatalf("%v.FullDuplexCall(_) = _, %v", tc, err)
	}
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, 27182)
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
}

// DoComputeEngineCreds performs a unary RPC with compute engine auth.
func DoComputeEngineCreds(tc testpb.TestServiceClient, serviceAccount, oauthScope string) {
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
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
	if user != serviceAccount {
		grpclog.Fatalf("Got user name %q, want %q.", user, serviceAccount)
	}
	if !strings.Contains(oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, oauthScope)
	}
}

func getServiceAccountJSONKey(keyFile string) []byte {
	jsonKey, err := ioutil.ReadFile(keyFile)
	if err != nil {
		grpclog.Fatalf("Failed to read the service account key file: %v", err)
	}
	return jsonKey
}

// DoServiceAccountCreds performs a unary RPC with service account auth.
func DoServiceAccountCreds(tc testpb.TestServiceClient, serviceAccountKeyFile, oauthScope string) {
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
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
	jsonKey := getServiceAccountJSONKey(serviceAccountKeyFile)
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	if !strings.Contains(oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, oauthScope)
	}
}

// DoJWTTokenCreds performs a unary RPC with JWT token auth.
func DoJWTTokenCreds(tc testpb.TestServiceClient, serviceAccountKeyFile string) {
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
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
	jsonKey := getServiceAccountJSONKey(serviceAccountKeyFile)
	user := reply.GetUsername()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
}

// GetToken obtains an OAUTH token from the input.
func GetToken(serviceAccountKeyFile string, oauthScope string) *oauth2.Token {
	jsonKey := getServiceAccountJSONKey(serviceAccountKeyFile)
	config, err := google.JWTConfigFromJSON(jsonKey, oauthScope)
	if err != nil {
		grpclog.Fatalf("Failed to get the config: %v", err)
	}
	token, err := config.TokenSource(context.Background()).Token()
	if err != nil {
		grpclog.Fatalf("Failed to get the token: %v", err)
	}
	return token
}

// DoOauth2TokenCreds performs a unary RPC with OAUTH2 token auth.
func DoOauth2TokenCreds(tc testpb.TestServiceClient, serviceAccountKeyFile, oauthScope string) {
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
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
	jsonKey := getServiceAccountJSONKey(serviceAccountKeyFile)
	user := reply.GetUsername()
	scope := reply.GetOauthScope()
	if !strings.Contains(string(jsonKey), user) {
		grpclog.Fatalf("Got user name %q which is NOT a substring of %q.", user, jsonKey)
	}
	if !strings.Contains(oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, oauthScope)
	}
}

// DoPerRPCCreds performs a unary RPC with per RPC OAUTH2 token.
func DoPerRPCCreds(tc testpb.TestServiceClient, serviceAccountKeyFile, oauthScope string) {
	jsonKey := getServiceAccountJSONKey(serviceAccountKeyFile)
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, largeReqSize)
	req := &testpb.SimpleRequest{
		ResponseType:   testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize:   proto.Int32(int32(largeRespSize)),
		Payload:        pl,
		FillUsername:   proto.Bool(true),
		FillOauthScope: proto.Bool(true),
	}
	token := GetToken(serviceAccountKeyFile, oauthScope)
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
	if !strings.Contains(oauthScope, scope) {
		grpclog.Fatalf("Got OAuth scope %q which is NOT a substring of %q.", scope, oauthScope)
	}
}

var (
	testMetadata = metadata.MD{
		"key1": []string{"value1"},
		"key2": []string{"value2"},
	}
)

// DoCancelAfterBegin cancels the RPC after metadata has been sent but before payloads are sent.
func DoCancelAfterBegin(tc testpb.TestServiceClient) {
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
}

// DoCancelAfterFirstResponse cancels the RPC after receiving the first message from the server.
func DoCancelAfterFirstResponse(tc testpb.TestServiceClient) {
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
	pl := clientNewPayload(testpb.PayloadType_COMPRESSABLE, 27182)
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
}

type testServer struct {
}

// NewTestServer creates a test server for test service.
func NewTestServer() testpb.TestServiceServer {
	return &testServer{}
}

func (s *testServer) EmptyCall(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
	return new(testpb.Empty), nil
}

func serverNewPayload(t testpb.PayloadType, size int32) (*testpb.Payload, error) {
	if size < 0 {
		return nil, fmt.Errorf("requested a response with invalid length %d", size)
	}
	body := make([]byte, size)
	switch t {
	case testpb.PayloadType_COMPRESSABLE:
	case testpb.PayloadType_UNCOMPRESSABLE:
		return nil, fmt.Errorf("payloadType UNCOMPRESSABLE is not supported")
	default:
		return nil, fmt.Errorf("unsupported payload type: %d", t)
	}
	return &testpb.Payload{
		Type: t.Enum(),
		Body: body,
	}, nil
}

func (s *testServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	pl, err := serverNewPayload(in.GetResponseType(), in.GetResponseSize())
	if err != nil {
		return nil, err
	}
	return &testpb.SimpleResponse{
		Payload: pl,
	}, nil
}

func (s *testServer) StreamingOutputCall(args *testpb.StreamingOutputCallRequest, stream testpb.TestService_StreamingOutputCallServer) error {
	cs := args.GetResponseParameters()
	for _, c := range cs {
		if us := c.GetIntervalUs(); us > 0 {
			time.Sleep(time.Duration(us) * time.Microsecond)
		}
		pl, err := serverNewPayload(args.GetResponseType(), c.GetSize())
		if err != nil {
			return err
		}
		if err := stream.Send(&testpb.StreamingOutputCallResponse{
			Payload: pl,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *testServer) StreamingInputCall(stream testpb.TestService_StreamingInputCallServer) error {
	var sum int
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&testpb.StreamingInputCallResponse{
				AggregatedPayloadSize: proto.Int32(int32(sum)),
			})
		}
		if err != nil {
			return err
		}
		p := in.GetPayload().GetBody()
		sum += len(p)
	}
}

func (s *testServer) FullDuplexCall(stream testpb.TestService_FullDuplexCallServer) error {
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			return nil
		}
		if err != nil {
			return err
		}
		cs := in.GetResponseParameters()
		for _, c := range cs {
			if us := c.GetIntervalUs(); us > 0 {
				time.Sleep(time.Duration(us) * time.Microsecond)
			}
			pl, err := serverNewPayload(in.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}
			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: pl,
			}); err != nil {
				return err
			}
		}
	}
}

func (s *testServer) HalfDuplexCall(stream testpb.TestService_HalfDuplexCallServer) error {
	var msgBuf []*testpb.StreamingOutputCallRequest
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			break
		}
		if err != nil {
			return err
		}
		msgBuf = append(msgBuf, in)
	}
	for _, m := range msgBuf {
		cs := m.GetResponseParameters()
		for _, c := range cs {
			if us := c.GetIntervalUs(); us > 0 {
				time.Sleep(time.Duration(us) * time.Microsecond)
			}
			pl, err := serverNewPayload(m.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}
			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: pl,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}
