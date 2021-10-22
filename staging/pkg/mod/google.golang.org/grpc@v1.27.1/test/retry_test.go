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

package test

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	testpb "google.golang.org/grpc/test/grpc_testing"
)

func enableRetry() func() {
	old := envconfig.Retry
	envconfig.Retry = true
	return func() { envconfig.Retry = old }
}

func (s) TestRetryUnary(t *testing.T) {
	defer enableRetry()()
	i := -1
	ss := &stubServer{
		emptyCall: func(context.Context, *testpb.Empty) (*testpb.Empty, error) {
			i++
			switch i {
			case 0, 2, 5:
				return &testpb.Empty{}, nil
			case 6, 8, 11:
				return nil, status.New(codes.Internal, "non-retryable error").Err()
			}
			return nil, status.New(codes.AlreadyExists, "retryable error").Err()
		},
	}
	if err := ss.Start([]grpc.ServerOption{}); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()
	ss.newServiceConfig(`{
    "methodConfig": [{
      "name": [{"service": "grpc.testing.TestService"}],
      "waitForReady": true,
      "retryPolicy": {
        "MaxAttempts": 4,
        "InitialBackoff": ".01s",
        "MaxBackoff": ".01s",
        "BackoffMultiplier": 1.0,
        "RetryableStatusCodes": [ "ALREADY_EXISTS" ]
      }
    }]}`)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timed out waiting for service config update")
		}
		if ss.cc.GetMethodConfig("/grpc.testing.TestService/EmptyCall").WaitForReady != nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	testCases := []struct {
		code  codes.Code
		count int
	}{
		{codes.OK, 0},
		{codes.OK, 2},
		{codes.OK, 5},
		{codes.Internal, 6},
		{codes.Internal, 8},
		{codes.Internal, 11},
		{codes.AlreadyExists, 15},
	}
	for _, tc := range testCases {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, err := ss.client.EmptyCall(ctx, &testpb.Empty{})
		cancel()
		if status.Code(err) != tc.code {
			t.Fatalf("EmptyCall(_, _) = _, %v; want _, <Code() = %v>", err, tc.code)
		}
		if i != tc.count {
			t.Fatalf("i = %v; want %v", i, tc.count)
		}
	}
}

func (s) TestRetryDisabledByDefault(t *testing.T) {
	if strings.EqualFold(os.Getenv("GRPC_GO_RETRY"), "on") {
		return
	}
	i := -1
	ss := &stubServer{
		emptyCall: func(context.Context, *testpb.Empty) (*testpb.Empty, error) {
			i++
			switch i {
			case 0:
				return nil, status.New(codes.AlreadyExists, "retryable error").Err()
			}
			return &testpb.Empty{}, nil
		},
	}
	if err := ss.Start([]grpc.ServerOption{}); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()
	ss.newServiceConfig(`{
    "methodConfig": [{
      "name": [{"service": "grpc.testing.TestService"}],
      "waitForReady": true,
      "retryPolicy": {
        "MaxAttempts": 4,
        "InitialBackoff": ".01s",
        "MaxBackoff": ".01s",
        "BackoffMultiplier": 1.0,
        "RetryableStatusCodes": [ "ALREADY_EXISTS" ]
      }
    }]}`)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timed out waiting for service config update")
		}
		if ss.cc.GetMethodConfig("/grpc.testing.TestService/EmptyCall").WaitForReady != nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	testCases := []struct {
		code  codes.Code
		count int
	}{
		{codes.AlreadyExists, 0},
	}
	for _, tc := range testCases {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, err := ss.client.EmptyCall(ctx, &testpb.Empty{})
		cancel()
		if status.Code(err) != tc.code {
			t.Fatalf("EmptyCall(_, _) = _, %v; want _, <Code() = %v>", err, tc.code)
		}
		if i != tc.count {
			t.Fatalf("i = %v; want %v", i, tc.count)
		}
	}
}

func (s) TestRetryThrottling(t *testing.T) {
	defer enableRetry()()
	i := -1
	ss := &stubServer{
		emptyCall: func(context.Context, *testpb.Empty) (*testpb.Empty, error) {
			i++
			switch i {
			case 0, 3, 6, 10, 11, 12, 13, 14, 16, 18:
				return &testpb.Empty{}, nil
			}
			return nil, status.New(codes.Unavailable, "retryable error").Err()
		},
	}
	if err := ss.Start([]grpc.ServerOption{}); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()
	ss.newServiceConfig(`{
    "methodConfig": [{
      "name": [{"service": "grpc.testing.TestService"}],
      "waitForReady": true,
      "retryPolicy": {
        "MaxAttempts": 4,
        "InitialBackoff": ".01s",
        "MaxBackoff": ".01s",
        "BackoffMultiplier": 1.0,
        "RetryableStatusCodes": [ "UNAVAILABLE" ]
      }
    }],
    "retryThrottling": {
      "maxTokens": 10,
      "tokenRatio": 0.5
    }
  }`)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timed out waiting for service config update")
		}
		if ss.cc.GetMethodConfig("/grpc.testing.TestService/EmptyCall").WaitForReady != nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	testCases := []struct {
		code  codes.Code
		count int
	}{
		{codes.OK, 0},           // tokens = 10
		{codes.OK, 3},           // tokens = 8.5 (10 - 2 failures + 0.5 success)
		{codes.OK, 6},           // tokens = 6
		{codes.Unavailable, 8},  // tokens = 5 -- first attempt is retried; second aborted.
		{codes.Unavailable, 9},  // tokens = 4
		{codes.OK, 10},          // tokens = 4.5
		{codes.OK, 11},          // tokens = 5
		{codes.OK, 12},          // tokens = 5.5
		{codes.OK, 13},          // tokens = 6
		{codes.OK, 14},          // tokens = 6.5
		{codes.OK, 16},          // tokens = 5.5
		{codes.Unavailable, 17}, // tokens = 4.5
	}
	for _, tc := range testCases {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, err := ss.client.EmptyCall(ctx, &testpb.Empty{})
		cancel()
		if status.Code(err) != tc.code {
			t.Errorf("EmptyCall(_, _) = _, %v; want _, <Code() = %v>", err, tc.code)
		}
		if i != tc.count {
			t.Errorf("i = %v; want %v", i, tc.count)
		}
	}
}

func (s) TestRetryStreaming(t *testing.T) {
	defer enableRetry()()
	req := func(b byte) *testpb.StreamingOutputCallRequest {
		return &testpb.StreamingOutputCallRequest{Payload: &testpb.Payload{Body: []byte{b}}}
	}
	res := func(b byte) *testpb.StreamingOutputCallResponse {
		return &testpb.StreamingOutputCallResponse{Payload: &testpb.Payload{Body: []byte{b}}}
	}

	largePayload, _ := newPayload(testpb.PayloadType_COMPRESSABLE, 500)

	type serverOp func(stream testpb.TestService_FullDuplexCallServer) error
	type clientOp func(stream testpb.TestService_FullDuplexCallClient) error

	// Server Operations
	sAttempts := func(n int) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			const key = "grpc-previous-rpc-attempts"
			md, ok := metadata.FromIncomingContext(stream.Context())
			if !ok {
				return status.Errorf(codes.Internal, "server: no header metadata received")
			}
			if got := md[key]; len(got) != 1 || got[0] != strconv.Itoa(n) {
				return status.Errorf(codes.Internal, "server: metadata = %v; want <contains %q: %q>", md, key, n)
			}
			return nil
		}
	}
	sReq := func(b byte) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			want := req(b)
			if got, err := stream.Recv(); err != nil || !proto.Equal(got, want) {
				return status.Errorf(codes.Internal, "server: Recv() = %v, %v; want %v, <nil>", got, err, want)
			}
			return nil
		}
	}
	sReqPayload := func(p *testpb.Payload) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			want := &testpb.StreamingOutputCallRequest{Payload: p}
			if got, err := stream.Recv(); err != nil || !proto.Equal(got, want) {
				return status.Errorf(codes.Internal, "server: Recv() = %v, %v; want %v, <nil>", got, err, want)
			}
			return nil
		}
	}
	sRes := func(b byte) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			msg := res(b)
			if err := stream.Send(msg); err != nil {
				return status.Errorf(codes.Internal, "server: Send(%v) = %v; want <nil>", msg, err)
			}
			return nil
		}
	}
	sErr := func(c codes.Code) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			return status.New(c, "").Err()
		}
	}
	sCloseSend := func() serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			if msg, err := stream.Recv(); msg != nil || err != io.EOF {
				return status.Errorf(codes.Internal, "server: Recv() = %v, %v; want <nil>, io.EOF", msg, err)
			}
			return nil
		}
	}
	sPushback := func(s string) serverOp {
		return func(stream testpb.TestService_FullDuplexCallServer) error {
			stream.SetTrailer(metadata.MD{"grpc-retry-pushback-ms": []string{s}})
			return nil
		}
	}

	// Client Operations
	cReq := func(b byte) clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			msg := req(b)
			if err := stream.Send(msg); err != nil {
				return fmt.Errorf("client: Send(%v) = %v; want <nil>", msg, err)
			}
			return nil
		}
	}
	cReqPayload := func(p *testpb.Payload) clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			msg := &testpb.StreamingOutputCallRequest{Payload: p}
			if err := stream.Send(msg); err != nil {
				return fmt.Errorf("client: Send(%v) = %v; want <nil>", msg, err)
			}
			return nil
		}
	}
	cRes := func(b byte) clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			want := res(b)
			if got, err := stream.Recv(); err != nil || !proto.Equal(got, want) {
				return fmt.Errorf("client: Recv() = %v, %v; want %v, <nil>", got, err, want)
			}
			return nil
		}
	}
	cErr := func(c codes.Code) clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			want := status.New(c, "").Err()
			if c == codes.OK {
				want = io.EOF
			}
			res, err := stream.Recv()
			if res != nil ||
				((err == nil) != (want == nil)) ||
				(want != nil && err.Error() != want.Error()) {
				return fmt.Errorf("client: Recv() = %v, %v; want <nil>, %v", res, err, want)
			}
			return nil
		}
	}
	cCloseSend := func() clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			if err := stream.CloseSend(); err != nil {
				return fmt.Errorf("client: CloseSend() = %v; want <nil>", err)
			}
			return nil
		}
	}
	var curTime time.Time
	cGetTime := func() clientOp {
		return func(_ testpb.TestService_FullDuplexCallClient) error {
			curTime = time.Now()
			return nil
		}
	}
	cCheckElapsed := func(d time.Duration) clientOp {
		return func(_ testpb.TestService_FullDuplexCallClient) error {
			if elapsed := time.Since(curTime); elapsed < d {
				return fmt.Errorf("elapsed time: %v; want >= %v", elapsed, d)
			}
			return nil
		}
	}
	cHdr := func() clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			_, err := stream.Header()
			return err
		}
	}
	cCtx := func() clientOp {
		return func(stream testpb.TestService_FullDuplexCallClient) error {
			stream.Context()
			return nil
		}
	}

	testCases := []struct {
		desc      string
		serverOps []serverOp
		clientOps []clientOp
	}{{
		desc:      "Non-retryable error code",
		serverOps: []serverOp{sReq(1), sErr(codes.Internal)},
		clientOps: []clientOp{cReq(1), cErr(codes.Internal)},
	}, {
		desc:      "One retry necessary",
		serverOps: []serverOp{sReq(1), sErr(codes.Unavailable), sReq(1), sAttempts(1), sRes(1)},
		clientOps: []clientOp{cReq(1), cRes(1), cErr(codes.OK)},
	}, {
		desc: "Exceed max attempts (4); check attempts header on server",
		serverOps: []serverOp{
			sReq(1), sErr(codes.Unavailable),
			sReq(1), sAttempts(1), sErr(codes.Unavailable),
			sAttempts(2), sReq(1), sErr(codes.Unavailable),
			sAttempts(3), sReq(1), sErr(codes.Unavailable),
		},
		clientOps: []clientOp{cReq(1), cErr(codes.Unavailable)},
	}, {
		desc: "Multiple requests",
		serverOps: []serverOp{
			sReq(1), sReq(2), sErr(codes.Unavailable),
			sReq(1), sReq(2), sRes(5),
		},
		clientOps: []clientOp{cReq(1), cReq(2), cRes(5), cErr(codes.OK)},
	}, {
		desc: "Multiple successive requests",
		serverOps: []serverOp{
			sReq(1), sErr(codes.Unavailable),
			sReq(1), sReq(2), sErr(codes.Unavailable),
			sReq(1), sReq(2), sReq(3), sRes(5),
		},
		clientOps: []clientOp{cReq(1), cReq(2), cReq(3), cRes(5), cErr(codes.OK)},
	}, {
		desc: "No retry after receiving",
		serverOps: []serverOp{
			sReq(1), sErr(codes.Unavailable),
			sReq(1), sRes(3), sErr(codes.Unavailable),
		},
		clientOps: []clientOp{cReq(1), cRes(3), cErr(codes.Unavailable)},
	}, {
		desc:      "No retry after header",
		serverOps: []serverOp{sReq(1), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReq(1), cHdr(), cErr(codes.Unavailable)},
	}, {
		desc:      "No retry after context",
		serverOps: []serverOp{sReq(1), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReq(1), cCtx(), cErr(codes.Unavailable)},
	}, {
		desc: "Replaying close send",
		serverOps: []serverOp{
			sReq(1), sReq(2), sCloseSend(), sErr(codes.Unavailable),
			sReq(1), sReq(2), sCloseSend(), sRes(1), sRes(3), sRes(5),
		},
		clientOps: []clientOp{cReq(1), cReq(2), cCloseSend(), cRes(1), cRes(3), cRes(5), cErr(codes.OK)},
	}, {
		desc:      "Negative server pushback - no retry",
		serverOps: []serverOp{sReq(1), sPushback("-1"), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReq(1), cErr(codes.Unavailable)},
	}, {
		desc:      "Non-numeric server pushback - no retry",
		serverOps: []serverOp{sReq(1), sPushback("xxx"), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReq(1), cErr(codes.Unavailable)},
	}, {
		desc:      "Multiple server pushback values - no retry",
		serverOps: []serverOp{sReq(1), sPushback("100"), sPushback("10"), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReq(1), cErr(codes.Unavailable)},
	}, {
		desc:      "1s server pushback - delayed retry",
		serverOps: []serverOp{sReq(1), sPushback("1000"), sErr(codes.Unavailable), sReq(1), sRes(2)},
		clientOps: []clientOp{cGetTime(), cReq(1), cRes(2), cCheckElapsed(time.Second), cErr(codes.OK)},
	}, {
		desc:      "Overflowing buffer - no retry",
		serverOps: []serverOp{sReqPayload(largePayload), sErr(codes.Unavailable)},
		clientOps: []clientOp{cReqPayload(largePayload), cErr(codes.Unavailable)},
	}}

	var serverOpIter int
	var serverOps []serverOp
	ss := &stubServer{
		fullDuplexCall: func(stream testpb.TestService_FullDuplexCallServer) error {
			for serverOpIter < len(serverOps) {
				op := serverOps[serverOpIter]
				serverOpIter++
				if err := op(stream); err != nil {
					return err
				}
			}
			return nil
		},
	}
	if err := ss.Start([]grpc.ServerOption{}, grpc.WithDefaultCallOptions(grpc.MaxRetryRPCBufferSize(200))); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer ss.Stop()
	ss.newServiceConfig(`{
    "methodConfig": [{
      "name": [{"service": "grpc.testing.TestService"}],
      "waitForReady": true,
      "retryPolicy": {
          "MaxAttempts": 4,
          "InitialBackoff": ".01s",
          "MaxBackoff": ".01s",
          "BackoffMultiplier": 1.0,
          "RetryableStatusCodes": [ "UNAVAILABLE" ]
      }
    }]}`)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	for {
		if ctx.Err() != nil {
			t.Fatalf("Timed out waiting for service config update")
		}
		if ss.cc.GetMethodConfig("/grpc.testing.TestService/FullDuplexCall").WaitForReady != nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	for _, tc := range testCases {
		func() {
			serverOpIter = 0
			serverOps = tc.serverOps

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			stream, err := ss.client.FullDuplexCall(ctx)
			if err != nil {
				t.Fatalf("%v: Error while creating stream: %v", tc.desc, err)
			}
			for _, op := range tc.clientOps {
				if err := op(stream); err != nil {
					t.Errorf("%v: %v", tc.desc, err)
					break
				}
			}
			if serverOpIter != len(serverOps) {
				t.Errorf("%v: serverOpIter = %v; want %v", tc.desc, serverOpIter, len(serverOps))
			}
		}()
	}
}
