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

package grpc_test

import (
	"fmt"
	"io"
	"math"
	"net"
	"reflect"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1alpha"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	testpb "google.golang.org/grpc/test/grpc_testing"
)

var (
	testMetadata = metadata.MD{
		"key1": []string{"value1"},
		"key2": []string{"value2"},
	}
	testAppUA = "myApp1/1.0 myApp2/0.9"
)

type testServer struct {
	security string // indicate the authentication protocol used by this server.
}

func (s *testServer) EmptyCall(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
	if md, ok := metadata.FromContext(ctx); ok {
		// For testing purpose, returns an error if there is attached metadata other than
		// the user agent set by the client application.
		if _, ok := md["user-agent"]; !ok {
			return nil, grpc.Errorf(codes.DataLoss, "got extra metadata")
		}
		var str []string
		for _, entry := range md["user-agent"] {
			str = append(str, "ua", entry)
		}
		grpc.SendHeader(ctx, metadata.Pairs(str...))
	}
	return new(testpb.Empty), nil
}

func newPayload(t testpb.PayloadType, size int32) (*testpb.Payload, error) {
	if size < 0 {
		return nil, fmt.Errorf("Requested a response with invalid length %d", size)
	}
	body := make([]byte, size)
	switch t {
	case testpb.PayloadType_COMPRESSABLE:
	case testpb.PayloadType_UNCOMPRESSABLE:
		return nil, fmt.Errorf("PayloadType UNCOMPRESSABLE is not supported")
	default:
		return nil, fmt.Errorf("Unsupported payload type: %d", t)
	}
	return &testpb.Payload{
		Type: t.Enum(),
		Body: body,
	}, nil
}

func (s *testServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	md, ok := metadata.FromContext(ctx)
	if ok {
		if err := grpc.SendHeader(ctx, md); err != nil {
			return nil, fmt.Errorf("grpc.SendHeader(%v, %v) = %v, want %v", ctx, md, err, nil)
		}
		grpc.SetTrailer(ctx, md)
	}
	pr, ok := peer.FromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("failed to get peer from ctx")
	}
	if pr.Addr == net.Addr(nil) {
		return nil, fmt.Errorf("failed to get peer address")
	}
	if s.security != "" {
		// Check Auth info
		var authType, serverName string
		switch info := pr.AuthInfo.(type) {
		case credentials.TLSInfo:
			authType = info.AuthType()
			serverName = info.State.ServerName
		default:
			return nil, fmt.Errorf("Unknown AuthInfo type")
		}
		if authType != s.security {
			return nil, fmt.Errorf("Wrong auth type: got %q, want %q", authType, s.security)
		}
		if serverName != "x.test.youtube.com" {
			return nil, fmt.Errorf("Unknown server name %q", serverName)
		}
	}

	// Simulate some service delay.
	time.Sleep(time.Second)

	payload, err := newPayload(in.GetResponseType(), in.GetResponseSize())
	if err != nil {
		return nil, err
	}
	return &testpb.SimpleResponse{
		Payload: payload,
	}, nil
}

func (s *testServer) StreamingOutputCall(args *testpb.StreamingOutputCallRequest, stream testpb.TestService_StreamingOutputCallServer) error {
	if md, ok := metadata.FromContext(stream.Context()); ok {
		// For testing purpose, returns an error if there is attached metadata.
		if len(md) > 0 {
			return grpc.Errorf(codes.DataLoss, "got extra metadata")
		}
	}
	cs := args.GetResponseParameters()
	for _, c := range cs {
		if us := c.GetIntervalUs(); us > 0 {
			time.Sleep(time.Duration(us) * time.Microsecond)
		}

		payload, err := newPayload(args.GetResponseType(), c.GetSize())
		if err != nil {
			return err
		}

		if err := stream.Send(&testpb.StreamingOutputCallResponse{
			Payload: payload,
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
	md, ok := metadata.FromContext(stream.Context())
	if ok {
		if err := stream.SendHeader(md); err != nil {
			return fmt.Errorf("%v.SendHeader(%v) = %v, want %v", stream, md, err, nil)
		}
		stream.SetTrailer(md)
	}
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

			payload, err := newPayload(in.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}

			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: payload,
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

			payload, err := newPayload(m.GetResponseType(), c.GetSize())
			if err != nil {
				return err
			}

			if err := stream.Send(&testpb.StreamingOutputCallResponse{
				Payload: payload,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

const tlsDir = "testdata/"

func TestReconnectTimeout(t *testing.T) {
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	_, port, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("Failed to parse listener address: %v", err)
	}
	addr := "localhost:" + port
	conn, err := grpc.Dial(addr, grpc.WithTimeout(5*time.Second), grpc.WithBlock(), grpc.WithInsecure())
	if err != nil {
		t.Fatalf("Failed to dial to the server %q: %v", addr, err)
	}
	// Close unaccepted connection (i.e., conn).
	lis.Close()
	tc := testpb.NewTestServiceClient(conn)
	waitC := make(chan struct{})
	go func() {
		defer close(waitC)
		const argSize = 271828
		const respSize = 314159

		payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
		if err != nil {
			t.Error(err)
			return
		}

		req := &testpb.SimpleRequest{
			ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
			ResponseSize: proto.Int32(respSize),
			Payload:      payload,
		}
		if _, err := tc.UnaryCall(context.Background(), req); err == nil {
			t.Errorf("TestService/UnaryCall(_, _) = _, <nil>, want _, non-nil")
			return
		}
	}()
	// Block untill reconnect times out.
	<-waitC
	if err := conn.Close(); err != grpc.ErrClientConnClosing {
		t.Fatalf("%v.Close() = %v, want %v", conn, err, grpc.ErrClientConnClosing)
	}
}

func unixDialer(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout("unix", addr, timeout)
}

type env struct {
	network  string // The type of network such as tcp, unix, etc.
	dialer   func(addr string, timeout time.Duration) (net.Conn, error)
	security string // The security protocol such as TLS, SSH, etc.
}

func listTestEnv() []env {
	if runtime.GOOS == "windows" {
		return []env{{"tcp", nil, ""}, {"tcp", nil, "tls"}}
	}
	return []env{{"tcp", nil, ""}, {"tcp", nil, "tls"}, {"unix", unixDialer, ""}, {"unix", unixDialer, "tls"}}
}

func serverSetUp(t *testing.T, servON bool, hs *health.HealthServer, maxStream uint32, cp grpc.Compressor, dc grpc.Decompressor, e env) (s *grpc.Server, addr string) {
	sopts := []grpc.ServerOption{grpc.MaxConcurrentStreams(maxStream), grpc.RPCCompressor(cp), grpc.RPCDecompressor(dc)}
	la := ":0"
	switch e.network {
	case "unix":
		la = "/tmp/testsock" + fmt.Sprintf("%d", time.Now())
		syscall.Unlink(la)
	}
	lis, err := net.Listen(e.network, la)
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	if e.security == "tls" {
		creds, err := credentials.NewServerTLSFromFile(tlsDir+"server1.pem", tlsDir+"server1.key")
		if err != nil {
			t.Fatalf("Failed to generate credentials %v", err)
		}
		sopts = append(sopts, grpc.Creds(creds))
	}
	s = grpc.NewServer(sopts...)
	if hs != nil {
		healthpb.RegisterHealthServer(s, hs)
	}
	if servON {
		testpb.RegisterTestServiceServer(s, &testServer{security: e.security})
	}
	go s.Serve(lis)
	addr = la
	switch e.network {
	case "unix":
	default:
		_, port, err := net.SplitHostPort(lis.Addr().String())
		if err != nil {
			t.Fatalf("Failed to parse listener address: %v", err)
		}
		addr = "localhost:" + port
	}
	return
}

func clientSetUp(t *testing.T, addr string, cp grpc.Compressor, dc grpc.Decompressor, ua string, e env) (cc *grpc.ClientConn) {
	var derr error
	if e.security == "tls" {
		creds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", "x.test.youtube.com")
		if err != nil {
			t.Fatalf("Failed to create credentials %v", err)
		}
		cc, derr = grpc.Dial(addr, grpc.WithTransportCredentials(creds), grpc.WithDialer(e.dialer), grpc.WithUserAgent(ua), grpc.WithCompressor(cp), grpc.WithDecompressor(dc))
	} else {
		cc, derr = grpc.Dial(addr, grpc.WithDialer(e.dialer), grpc.WithInsecure(), grpc.WithUserAgent(ua), grpc.WithCompressor(cp), grpc.WithDecompressor(dc))
	}
	if derr != nil {
		t.Fatalf("Dial(%q) = %v", addr, derr)
	}
	return
}

func tearDown(s *grpc.Server, cc *grpc.ClientConn) {
	cc.Close()
	s.Stop()
}

func TestTimeoutOnDeadServer(t *testing.T) {
	for _, e := range listTestEnv() {
		testTimeoutOnDeadServer(t, e)
	}
}

func testTimeoutOnDeadServer(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	ctx, _ := context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Idle); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Idle, err)
	}
	ctx, _ = context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Connecting); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Connecting, err)
	}
	if state, err := cc.State(); err != nil || state != grpc.Ready {
		t.Fatalf("cc.State() = %s, %v, want %s, <nil>", state, err, grpc.Ready)
	}
	ctx, _ = context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Ready); err != context.DeadlineExceeded {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, %v", grpc.Ready, err, context.DeadlineExceeded)
	}
	s.Stop()
	// Set -1 as the timeout to make sure if transportMonitor gets error
	// notification in time the failure path of the 1st invoke of
	// ClientConn.wait hits the deadline exceeded error.
	ctx, _ = context.WithTimeout(context.Background(), -1)
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); grpc.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("TestService/EmptyCall(%v, _) = _, error %v, want _, error code: %d", ctx, err, codes.DeadlineExceeded)
	}
	ctx, _ = context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Ready); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Ready, err)
	}
	if state, err := cc.State(); err != nil || (state != grpc.Connecting && state != grpc.TransientFailure) {
		t.Fatalf("cc.State() = %s, %v, want %s or %s, <nil>", state, err, grpc.Connecting, grpc.TransientFailure)
	}
	cc.Close()
}

func healthCheck(t time.Duration, cc *grpc.ClientConn, serviceName string) (*healthpb.HealthCheckResponse, error) {
	ctx, _ := context.WithTimeout(context.Background(), t)
	hc := healthpb.NewHealthClient(cc)
	req := &healthpb.HealthCheckRequest{
		Service: serviceName,
	}
	return hc.Check(ctx, req)
}

func TestHealthCheckOnSuccess(t *testing.T) {
	for _, e := range listTestEnv() {
		testHealthCheckOnSuccess(t, e)
	}
}

func testHealthCheckOnSuccess(t *testing.T, e env) {
	hs := health.NewHealthServer()
	hs.SetServingStatus("grpc.health.v1alpha.Health", 1)
	s, addr := serverSetUp(t, true, hs, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	defer tearDown(s, cc)
	if _, err := healthCheck(1*time.Second, cc, "grpc.health.v1alpha.Health"); err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
}

func TestHealthCheckOnFailure(t *testing.T) {
	for _, e := range listTestEnv() {
		testHealthCheckOnFailure(t, e)
	}
}

func testHealthCheckOnFailure(t *testing.T, e env) {
	hs := health.NewHealthServer()
	hs.SetServingStatus("grpc.health.v1alpha.HealthCheck", 1)
	s, addr := serverSetUp(t, true, hs, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	defer tearDown(s, cc)
	if _, err := healthCheck(0*time.Second, cc, "grpc.health.v1alpha.Health"); err != grpc.Errorf(codes.DeadlineExceeded, "context deadline exceeded") {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, error code %d", err, codes.DeadlineExceeded)
	}
}

func TestHealthCheckOff(t *testing.T) {
	for _, e := range listTestEnv() {
		testHealthCheckOff(t, e)
	}
}

func testHealthCheckOff(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	defer tearDown(s, cc)
	if _, err := healthCheck(1*time.Second, cc, ""); err != grpc.Errorf(codes.Unimplemented, "unknown service grpc.health.v1alpha.Health") {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, error code %d", err, codes.Unimplemented)
	}
}

func TestHealthCheckServingStatus(t *testing.T) {
	for _, e := range listTestEnv() {
		testHealthCheckServingStatus(t, e)
	}
}

func testHealthCheckServingStatus(t *testing.T, e env) {
	hs := health.NewHealthServer()
	s, addr := serverSetUp(t, true, hs, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	defer tearDown(s, cc)
	out, err := healthCheck(1*time.Second, cc, "")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("Got the serving status %v, want SERVING", out.Status)
	}
	if _, err := healthCheck(1*time.Second, cc, "grpc.health.v1alpha.Health"); err != grpc.Errorf(codes.NotFound, "unknown service") {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, error code %d", err, codes.NotFound)
	}
	hs.SetServingStatus("grpc.health.v1alpha.Health", healthpb.HealthCheckResponse_SERVING)
	out, err = healthCheck(1*time.Second, cc, "grpc.health.v1alpha.Health")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("Got the serving status %v, want SERVING", out.Status)
	}
	hs.SetServingStatus("grpc.health.v1alpha.Health", healthpb.HealthCheckResponse_NOT_SERVING)
	out, err = healthCheck(1*time.Second, cc, "grpc.health.v1alpha.Health")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_NOT_SERVING {
		t.Fatalf("Got the serving status %v, want NOT_SERVING", out.Status)
	}

}

func TestEmptyUnaryWithUserAgent(t *testing.T) {
	for _, e := range listTestEnv() {
		testEmptyUnaryWithUserAgent(t, e)
	}
}

func testEmptyUnaryWithUserAgent(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, testAppUA, e)
	// Wait until cc is connected.
	ctx, _ := context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Idle); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Idle, err)
	}
	ctx, _ = context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Connecting); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Connecting, err)
	}
	if state, err := cc.State(); err != nil || state != grpc.Ready {
		t.Fatalf("cc.State() = %s, %v, want %s, <nil>", state, err, grpc.Ready)
	}
	ctx, _ = context.WithTimeout(context.Background(), time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Ready); err == nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, <nil>, want _, %v", grpc.Ready, context.DeadlineExceeded)
	}
	tc := testpb.NewTestServiceClient(cc)
	var header metadata.MD
	reply, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.Header(&header))
	if err != nil || !proto.Equal(&testpb.Empty{}, reply) {
		t.Fatalf("TestService/EmptyCall(_, _) = %v, %v, want %v, <nil>", reply, err, &testpb.Empty{})
	}
	if v, ok := header["ua"]; !ok || v[0] != testAppUA {
		t.Fatalf("header[\"ua\"] = %q, %t, want %q, true", v, ok, testAppUA)
	}
	tearDown(s, cc)
	ctx, _ = context.WithTimeout(context.Background(), 5*time.Second)
	if _, err := cc.WaitForStateChange(ctx, grpc.Ready); err != nil {
		t.Fatalf("cc.WaitForStateChange(_, %s) = _, %v, want _, <nil>", grpc.Ready, err)
	}
	if state, err := cc.State(); err != nil || state != grpc.Shutdown {
		t.Fatalf("cc.State() = %s, %v, want %s, <nil>", state, err, grpc.Shutdown)
	}
}

func TestFailedEmptyUnary(t *testing.T) {
	for _, e := range listTestEnv() {
		testFailedEmptyUnary(t, e)
	}
}

func testFailedEmptyUnary(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	ctx := metadata.NewContext(context.Background(), testMetadata)
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); err != grpc.Errorf(codes.DataLoss, "got extra metadata") {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %v", err, grpc.Errorf(codes.DataLoss, "got extra metadata"))
	}
}

func TestLargeUnary(t *testing.T) {
	for _, e := range listTestEnv() {
		testLargeUnary(t, e)
	}
}

func testLargeUnary(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 271828
	respSize := 314159

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
	}
	pt := reply.GetPayload().GetType()
	ps := len(reply.GetPayload().GetBody())
	if pt != testpb.PayloadType_COMPRESSABLE || ps != respSize {
		t.Fatalf("Got the reply with type %d len %d; want %d, %d", pt, ps, testpb.PayloadType_COMPRESSABLE, respSize)
	}
}

func TestMetadataUnaryRPC(t *testing.T) {
	for _, e := range listTestEnv() {
		testMetadataUnaryRPC(t, e)
	}
}

func testMetadataUnaryRPC(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 2718
	respSize := 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	var header, trailer metadata.MD
	ctx := metadata.NewContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Header(&header), grpc.Trailer(&trailer)); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}
	if !reflect.DeepEqual(testMetadata, header) {
		t.Fatalf("Received header metadata %v, want %v", header, testMetadata)
	}
	if !reflect.DeepEqual(testMetadata, trailer) {
		t.Fatalf("Received trailer metadata %v, want %v", trailer, testMetadata)
	}
}

func performOneRPC(t *testing.T, tc testpb.TestServiceClient, wg *sync.WaitGroup) {
	defer wg.Done()
	const argSize = 2718
	const respSize = 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Error(err)
		return
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	reply, err := tc.UnaryCall(context.Background(), req)
	if err != nil {
		t.Errorf("TestService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
		return
	}
	pt := reply.GetPayload().GetType()
	ps := len(reply.GetPayload().GetBody())
	if pt != testpb.PayloadType_COMPRESSABLE || ps != respSize {
		t.Errorf("Got reply with type %d len %d; want %d, %d", pt, ps, testpb.PayloadType_COMPRESSABLE, respSize)
		return
	}
}

func TestRetry(t *testing.T) {
	for _, e := range listTestEnv() {
		testRetry(t, e)
	}
}

// This test mimics a user who sends 1000 RPCs concurrently on a faulty transport.
// TODO(zhaoq): Refactor to make this clearer and add more cases to test racy
// and error-prone paths.
func testRetry(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		time.Sleep(1 * time.Second)
		// The server shuts down the network connection to make a
		// transport error which will be detected by the client side
		// code.
		s.TestingCloseConns()
		wg.Done()
	}()
	// All these RPCs should succeed eventually.
	for i := 0; i < 1000; i++ {
		time.Sleep(2 * time.Millisecond)
		wg.Add(1)
		go performOneRPC(t, tc, &wg)
	}
	wg.Wait()
}

func TestRPCTimeout(t *testing.T) {
	for _, e := range listTestEnv() {
		testRPCTimeout(t, e)
	}
}

// TODO(zhaoq): Have a better test coverage of timeout and cancellation mechanism.
func testRPCTimeout(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 2718
	respSize := 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	for i := -1; i <= 10; i++ {
		ctx, _ := context.WithTimeout(context.Background(), time.Duration(i)*time.Millisecond)
		reply, err := tc.UnaryCall(ctx, req)
		if grpc.Code(err) != codes.DeadlineExceeded {
			t.Fatalf(`TestService/UnaryCallv(_, _) = %v, %v; want <nil>, error code: %d`, reply, err, codes.DeadlineExceeded)
		}
	}
}

func TestCancel(t *testing.T) {
	for _, e := range listTestEnv() {
		testCancel(t, e)
	}
}

func testCancel(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 2718
	respSize := 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(1*time.Millisecond, cancel)
	reply, err := tc.UnaryCall(ctx, req)
	if grpc.Code(err) != codes.Canceled {
		t.Fatalf(`TestService/UnaryCall(_, _) = %v, %v; want <nil>, error code: %d`, reply, err, codes.Canceled)
	}
}

func TestCancelNoIO(t *testing.T) {
	for _, e := range listTestEnv() {
		testCancelNoIO(t, e)
	}
}

func testCancelNoIO(t *testing.T, e env) {
	// Only allows 1 live stream per server transport.
	s, addr := serverSetUp(t, true, nil, 1, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	ctx, cancel := context.WithCancel(context.Background())
	_, err := tc.StreamingInputCall(ctx)
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
	}
	// Loop until receiving the new max stream setting from the server.
	for {
		ctx, _ := context.WithTimeout(context.Background(), time.Second)
		_, err := tc.StreamingInputCall(ctx)
		if err == nil {
			time.Sleep(time.Second)
			continue
		}
		if grpc.Code(err) == codes.DeadlineExceeded {
			break
		}
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %d", tc, err, codes.DeadlineExceeded)
	}
	// If there are any RPCs slipping before the client receives the max streams setting,
	// let them be expired.
	time.Sleep(2 * time.Second)
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		// This should be blocked until the 1st is canceled.
		ctx, _ := context.WithTimeout(context.Background(), 2*time.Second)
		if _, err := tc.StreamingInputCall(ctx); err != nil {
			t.Errorf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
		}
	}()
	cancel()
	<-ch
}

// The following tests the gRPC streaming RPC implementations.
// TODO(zhaoq): Have better coverage on error cases.
var (
	reqSizes  = []int{27182, 8, 1828, 45904}
	respSizes = []int{31415, 9, 2653, 58979}
)

func TestNoService(t *testing.T) {
	for _, e := range listTestEnv() {
		testNoService(t, e)
	}
}

func testNoService(t *testing.T, e env) {
	s, addr := serverSetUp(t, false, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	// Make sure setting ack has been sent.
	time.Sleep(2 * time.Second)
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	if _, err := stream.Recv(); grpc.Code(err) != codes.Unimplemented {
		t.Fatalf("stream.Recv() = _, %v, want _, error code %d", err, codes.Unimplemented)
	}
}

func TestPingPong(t *testing.T) {
	for _, e := range listTestEnv() {
		testPingPong(t, e)
	}
}

func testPingPong(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	var index int
	for index < len(reqSizes) {
		respParam := []*testpb.ResponseParameters{
			{
				Size: proto.Int32(int32(respSizes[index])),
			},
		}

		payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(reqSizes[index]))
		if err != nil {
			t.Fatal(err)
		}

		req := &testpb.StreamingOutputCallRequest{
			ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
			ResponseParameters: respParam,
			Payload:            payload,
		}
		if err := stream.Send(req); err != nil {
			t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
		}
		reply, err := stream.Recv()
		if err != nil {
			t.Fatalf("%v.Recv() = %v, want <nil>", stream, err)
		}
		pt := reply.GetPayload().GetType()
		if pt != testpb.PayloadType_COMPRESSABLE {
			t.Fatalf("Got the reply of type %d, want %d", pt, testpb.PayloadType_COMPRESSABLE)
		}
		size := len(reply.GetPayload().GetBody())
		if size != int(respSizes[index]) {
			t.Fatalf("Got reply body of length %d, want %d", size, respSizes[index])
		}
		index++
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("%v failed to complele the ping pong test: %v", stream, err)
	}
}

func TestMetadataStreamingRPC(t *testing.T) {
	for _, e := range listTestEnv() {
		testMetadataStreamingRPC(t, e)
	}
}

func testMetadataStreamingRPC(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	ctx := metadata.NewContext(context.Background(), testMetadata)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	go func() {
		headerMD, err := stream.Header()
		if e.security == "tls" {
			delete(headerMD, "transport_security_type")
		}
		if err != nil || !reflect.DeepEqual(testMetadata, headerMD) {
			t.Errorf("#1 %v.Header() = %v, %v, want %v, <nil>", stream, headerMD, err, testMetadata)
		}
		// test the cached value.
		headerMD, err = stream.Header()
		if err != nil || !reflect.DeepEqual(testMetadata, headerMD) {
			t.Errorf("#2 %v.Header() = %v, %v, want %v, <nil>", stream, headerMD, err, testMetadata)
		}
		var index int
		for index < len(reqSizes) {
			respParam := []*testpb.ResponseParameters{
				{
					Size: proto.Int32(int32(respSizes[index])),
				},
			}

			payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(reqSizes[index]))
			if err != nil {
				t.Fatal(err)
			}

			req := &testpb.StreamingOutputCallRequest{
				ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
				ResponseParameters: respParam,
				Payload:            payload,
			}
			if err := stream.Send(req); err != nil {
				t.Errorf("%v.Send(%v) = %v, want <nil>", stream, req, err)
				return
			}
			index++
		}
		// Tell the server we're done sending args.
		stream.CloseSend()
	}()
	for {
		if _, err := stream.Recv(); err != nil {
			break
		}
	}
	trailerMD := stream.Trailer()
	if !reflect.DeepEqual(testMetadata, trailerMD) {
		t.Fatalf("%v.Trailer() = %v, want %v", stream, trailerMD, testMetadata)
	}
}

func TestServerStreaming(t *testing.T) {
	for _, e := range listTestEnv() {
		testServerStreaming(t, e)
	}
}

func testServerStreaming(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
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
		t.Fatalf("%v.StreamingOutputCall(_) = _, %v, want <nil>", tc, err)
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
		pt := reply.GetPayload().GetType()
		if pt != testpb.PayloadType_COMPRESSABLE {
			t.Fatalf("Got the reply of type %d, want %d", pt, testpb.PayloadType_COMPRESSABLE)
		}
		size := len(reply.GetPayload().GetBody())
		if size != int(respSizes[index]) {
			t.Fatalf("Got reply body of length %d, want %d", size, respSizes[index])
		}
		index++
		respCnt++
	}
	if rpcStatus != io.EOF {
		t.Fatalf("Failed to finish the server streaming rpc: %v, want <EOF>", rpcStatus)
	}
	if respCnt != len(respSizes) {
		t.Fatalf("Got %d reply, want %d", len(respSizes), respCnt)
	}
}

func TestFailedServerStreaming(t *testing.T) {
	for _, e := range listTestEnv() {
		testFailedServerStreaming(t, e)
	}
}

func testFailedServerStreaming(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
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
	ctx := metadata.NewContext(context.Background(), testMetadata)
	stream, err := tc.StreamingOutputCall(ctx, req)
	if err != nil {
		t.Fatalf("%v.StreamingOutputCall(_) = _, %v, want <nil>", tc, err)
	}
	if _, err := stream.Recv(); err != grpc.Errorf(codes.DataLoss, "got extra metadata") {
		t.Fatalf("%v.Recv() = _, %v, want _, %v", stream, err, grpc.Errorf(codes.DataLoss, "got extra metadata"))
	}
}

func TestClientStreaming(t *testing.T) {
	for _, e := range listTestEnv() {
		testClientStreaming(t, e)
	}
}

func testClientStreaming(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	stream, err := tc.StreamingInputCall(context.Background())
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want <nil>", tc, err)
	}
	var sum int

	for _, s := range reqSizes {
		payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(s))
		if err != nil {
			t.Fatal(err)
		}

		req := &testpb.StreamingInputCallRequest{
			Payload: payload,
		}
		if err := stream.Send(req); err != nil {
			t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
		}
		sum += s
	}
	reply, err := stream.CloseAndRecv()
	if err != nil {
		t.Fatalf("%v.CloseAndRecv() got error %v, want %v", stream, err, nil)
	}
	if reply.GetAggregatedPayloadSize() != int32(sum) {
		t.Fatalf("%v.CloseAndRecv().GetAggregatePayloadSize() = %v; want %v", stream, reply.GetAggregatedPayloadSize(), sum)
	}
}

func TestExceedMaxStreamsLimit(t *testing.T) {
	for _, e := range listTestEnv() {
		testExceedMaxStreamsLimit(t, e)
	}
}

func testExceedMaxStreamsLimit(t *testing.T, e env) {
	// Only allows 1 live stream per server transport.
	s, addr := serverSetUp(t, true, nil, 1, nil, nil, e)
	cc := clientSetUp(t, addr, nil, nil, "", e)
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	_, err := tc.StreamingInputCall(context.Background())
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
	}
	// Loop until receiving the new max stream setting from the server.
	for {
		ctx, _ := context.WithTimeout(context.Background(), time.Second)
		_, err := tc.StreamingInputCall(ctx)
		if err == nil {
			time.Sleep(time.Second)
			continue
		}
		if grpc.Code(err) == codes.DeadlineExceeded {
			break
		}
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %d", tc, err, codes.DeadlineExceeded)
	}
}

func TestCompressServerHasNoSupport(t *testing.T) {
	for _, e := range listTestEnv() {
		testCompressServerHasNoSupport(t, e)
	}
}

func testCompressServerHasNoSupport(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, nil, nil, e)
	cc := clientSetUp(t, addr, grpc.NewGZIPCompressor(), nil, "", e)
	// Unary call
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 271828
	respSize := 314159
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || grpc.Code(err) != codes.InvalidArgument {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code %d", err, codes.InvalidArgument)
	}
	// Streaming RPC
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(31415),
		},
	}
	payload, err = newPayload(testpb.PayloadType_COMPRESSABLE, int32(31415))
	if err != nil {
		t.Fatal(err)
	}
	sreq := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err == nil || grpc.Code(err) != codes.InvalidArgument {
		t.Fatalf("%v.Recv() = %v, want error code %d", stream, err, codes.InvalidArgument)
	}
}

func TestCompressOK(t *testing.T) {
	for _, e := range listTestEnv() {
		testCompressOK(t, e)
	}
}

func testCompressOK(t *testing.T, e env) {
	s, addr := serverSetUp(t, true, nil, math.MaxUint32, grpc.NewGZIPCompressor(), grpc.NewGZIPDecompressor(), e)
	cc := clientSetUp(t, addr, grpc.NewGZIPCompressor(), grpc.NewGZIPDecompressor(), "", e)
	// Unary call
	tc := testpb.NewTestServiceClient(cc)
	defer tearDown(s, cc)
	argSize := 271828
	respSize := 314159
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(argSize))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(int32(respSize)),
		Payload:      payload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err != nil {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
	}
	// Streaming RPC
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(31415),
		},
	}
	payload, err = newPayload(testpb.PayloadType_COMPRESSABLE, int32(31415))
	if err != nil {
		t.Fatal(err)
	}
	sreq := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = %v, want <nil>", stream, err)
	}
}
