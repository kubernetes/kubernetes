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
	"bytes"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	anypb "github.com/golang/protobuf/ptypes/any"
	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/tap"
	testpb "google.golang.org/grpc/test/grpc_testing"
)

var (
	// For headers:
	testMetadata = metadata.MD{
		"key1":     []string{"value1"},
		"key2":     []string{"value2"},
		"key3-bin": []string{"binvalue1", string([]byte{1, 2, 3})},
	}
	testMetadata2 = metadata.MD{
		"key1": []string{"value12"},
		"key2": []string{"value22"},
	}
	// For trailers:
	testTrailerMetadata = metadata.MD{
		"tkey1":     []string{"trailerValue1"},
		"tkey2":     []string{"trailerValue2"},
		"tkey3-bin": []string{"trailerbinvalue1", string([]byte{3, 2, 1})},
	}
	testTrailerMetadata2 = metadata.MD{
		"tkey1": []string{"trailerValue12"},
		"tkey2": []string{"trailerValue22"},
	}
	// capital "Key" is illegal in HTTP/2.
	malformedHTTP2Metadata = metadata.MD{
		"Key": []string{"foo"},
	}
	testAppUA     = "myApp1/1.0 myApp2/0.9"
	failAppUA     = "fail-this-RPC"
	detailedError = status.ErrorProto(&spb.Status{
		Code:    int32(codes.DataLoss),
		Message: "error for testing: " + failAppUA,
		Details: []*anypb.Any{{
			TypeUrl: "url",
			Value:   []byte{6, 0, 0, 6, 1, 3},
		}},
	})
)

var raceMode bool // set by race_test.go in race mode

type testServer struct {
	security           string // indicate the authentication protocol used by this server.
	earlyFail          bool   // whether to error out the execution of a service handler prematurely.
	setAndSendHeader   bool   // whether to call setHeader and sendHeader.
	setHeaderOnly      bool   // whether to only call setHeader, not sendHeader.
	multipleSetTrailer bool   // whether to call setTrailer multiple times.
}

func (s *testServer) EmptyCall(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		// For testing purpose, returns an error if user-agent is failAppUA.
		// To test that client gets the correct error.
		if ua, ok := md["user-agent"]; !ok || strings.HasPrefix(ua[0], failAppUA) {
			return nil, detailedError
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
	md, ok := metadata.FromIncomingContext(ctx)
	if ok {
		if _, exists := md[":authority"]; !exists {
			return nil, grpc.Errorf(codes.DataLoss, "expected an :authority metadata: %v", md)
		}
		if s.setAndSendHeader {
			if err := grpc.SetHeader(ctx, md); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SetHeader(_, %v) = %v, want <nil>", md, err)
			}
			if err := grpc.SendHeader(ctx, testMetadata2); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SendHeader(_, %v) = %v, want <nil>", testMetadata2, err)
			}
		} else if s.setHeaderOnly {
			if err := grpc.SetHeader(ctx, md); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SetHeader(_, %v) = %v, want <nil>", md, err)
			}
			if err := grpc.SetHeader(ctx, testMetadata2); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SetHeader(_, %v) = %v, want <nil>", testMetadata2, err)
			}
		} else {
			if err := grpc.SendHeader(ctx, md); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SendHeader(_, %v) = %v, want <nil>", md, err)
			}
		}
		if err := grpc.SetTrailer(ctx, testTrailerMetadata); err != nil {
			return nil, grpc.Errorf(grpc.Code(err), "grpc.SetTrailer(_, %v) = %v, want <nil>", testTrailerMetadata, err)
		}
		if s.multipleSetTrailer {
			if err := grpc.SetTrailer(ctx, testTrailerMetadata2); err != nil {
				return nil, grpc.Errorf(grpc.Code(err), "grpc.SetTrailer(_, %v) = %v, want <nil>", testTrailerMetadata2, err)
			}
		}
	}
	pr, ok := peer.FromContext(ctx)
	if !ok {
		return nil, grpc.Errorf(codes.DataLoss, "failed to get peer from ctx")
	}
	if pr.Addr == net.Addr(nil) {
		return nil, grpc.Errorf(codes.DataLoss, "failed to get peer address")
	}
	if s.security != "" {
		// Check Auth info
		var authType, serverName string
		switch info := pr.AuthInfo.(type) {
		case credentials.TLSInfo:
			authType = info.AuthType()
			serverName = info.State.ServerName
		default:
			return nil, grpc.Errorf(codes.Unauthenticated, "Unknown AuthInfo type")
		}
		if authType != s.security {
			return nil, grpc.Errorf(codes.Unauthenticated, "Wrong auth type: got %q, want %q", authType, s.security)
		}
		if serverName != "x.test.youtube.com" {
			return nil, grpc.Errorf(codes.Unauthenticated, "Unknown server name %q", serverName)
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
	if md, ok := metadata.FromIncomingContext(stream.Context()); ok {
		if _, exists := md[":authority"]; !exists {
			return grpc.Errorf(codes.DataLoss, "expected an :authority metadata: %v", md)
		}
		// For testing purpose, returns an error if user-agent is failAppUA.
		// To test that client gets the correct error.
		if ua, ok := md["user-agent"]; !ok || strings.HasPrefix(ua[0], failAppUA) {
			return grpc.Errorf(codes.DataLoss, "error for testing: "+failAppUA)
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
		if s.earlyFail {
			return grpc.Errorf(codes.NotFound, "not found")
		}
	}
}

func (s *testServer) FullDuplexCall(stream testpb.TestService_FullDuplexCallServer) error {
	md, ok := metadata.FromIncomingContext(stream.Context())
	if ok {
		if s.setAndSendHeader {
			if err := stream.SetHeader(md); err != nil {
				return grpc.Errorf(grpc.Code(err), "%v.SetHeader(_, %v) = %v, want <nil>", stream, md, err)
			}
			if err := stream.SendHeader(testMetadata2); err != nil {
				return grpc.Errorf(grpc.Code(err), "%v.SendHeader(_, %v) = %v, want <nil>", stream, testMetadata2, err)
			}
		} else if s.setHeaderOnly {
			if err := stream.SetHeader(md); err != nil {
				return grpc.Errorf(grpc.Code(err), "%v.SetHeader(_, %v) = %v, want <nil>", stream, md, err)
			}
			if err := stream.SetHeader(testMetadata2); err != nil {
				return grpc.Errorf(grpc.Code(err), "%v.SetHeader(_, %v) = %v, want <nil>", stream, testMetadata2, err)
			}
		} else {
			if err := stream.SendHeader(md); err != nil {
				return grpc.Errorf(grpc.Code(err), "%v.SendHeader(%v) = %v, want %v", stream, md, err, nil)
			}
		}
		stream.SetTrailer(testTrailerMetadata)
		if s.multipleSetTrailer {
			stream.SetTrailer(testTrailerMetadata2)
		}
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

type env struct {
	name        string
	network     string // The type of network such as tcp, unix, etc.
	security    string // The security protocol such as TLS, SSH, etc.
	httpHandler bool   // whether to use the http.Handler ServerTransport; requires TLS
	balancer    bool   // whether to use balancer
}

func (e env) runnable() bool {
	if runtime.GOOS == "windows" && e.network == "unix" {
		return false
	}
	return true
}

func (e env) dialer(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout(e.network, addr, timeout)
}

var (
	tcpClearEnv   = env{name: "tcp-clear", network: "tcp", balancer: true}
	tcpTLSEnv     = env{name: "tcp-tls", network: "tcp", security: "tls", balancer: true}
	unixClearEnv  = env{name: "unix-clear", network: "unix", balancer: true}
	unixTLSEnv    = env{name: "unix-tls", network: "unix", security: "tls", balancer: true}
	handlerEnv    = env{name: "handler-tls", network: "tcp", security: "tls", httpHandler: true, balancer: true}
	noBalancerEnv = env{name: "no-balancer", network: "tcp", security: "tls", balancer: false}
	// TODO add handlerEnv back when ServeHTTP is stable.
	allEnv = []env{tcpClearEnv, tcpTLSEnv, unixClearEnv, unixTLSEnv /*handlerEnv,*/, noBalancerEnv}
)

var onlyEnv = flag.String("only_env", "", "If non-empty, one of 'tcp-clear', 'tcp-tls', 'unix-clear', 'unix-tls', or 'handler-tls' to only run the tests for that environment. Empty means all.")

func listTestEnv() (envs []env) {
	if *onlyEnv != "" {
		for _, e := range allEnv {
			if e.name == *onlyEnv {
				if !e.runnable() {
					panic(fmt.Sprintf("--only_env environment %q does not run on %s", *onlyEnv, runtime.GOOS))
				}
				return []env{e}
			}
		}
		panic(fmt.Sprintf("invalid --only_env value %q", *onlyEnv))
	}
	for _, e := range allEnv {
		if e.runnable() {
			envs = append(envs, e)
		}
	}
	return envs
}

// test is an end-to-end test. It should be created with the newTest
// func, modified as needed, and then started with its startServer method.
// It should be cleaned up with the tearDown method.
type test struct {
	t *testing.T
	e env

	ctx    context.Context // valid for life of test, before tearDown
	cancel context.CancelFunc

	// Configurable knobs, after newTest returns:
	testServer        testpb.TestServiceServer // nil means none
	healthServer      *health.Server           // nil means disabled
	maxStream         uint32
	tapHandle         tap.ServerInHandle
	maxMsgSize        int
	userAgent         string
	clientCompression bool
	serverCompression bool
	unaryClientInt    grpc.UnaryClientInterceptor
	streamClientInt   grpc.StreamClientInterceptor
	unaryServerInt    grpc.UnaryServerInterceptor
	streamServerInt   grpc.StreamServerInterceptor
	unknownHandler    grpc.StreamHandler
	sc                <-chan grpc.ServiceConfig

	// srv and srvAddr are set once startServer is called.
	srv     *grpc.Server
	srvAddr string

	cc          *grpc.ClientConn // nil until requested via clientConn
	restoreLogs func()           // nil unless declareLogNoise is used
}

func (te *test) tearDown() {
	if te.cancel != nil {
		te.cancel()
		te.cancel = nil
	}
	if te.cc != nil {
		te.cc.Close()
		te.cc = nil
	}
	if te.restoreLogs != nil {
		te.restoreLogs()
		te.restoreLogs = nil
	}
	if te.srv != nil {
		te.srv.Stop()
	}
}

// newTest returns a new test using the provided testing.T and
// environment.  It is returned with default values. Tests should
// modify it before calling its startServer and clientConn methods.
func newTest(t *testing.T, e env) *test {
	te := &test{
		t:         t,
		e:         e,
		maxStream: math.MaxUint32,
	}
	te.ctx, te.cancel = context.WithCancel(context.Background())
	return te
}

// startServer starts a gRPC server listening. Callers should defer a
// call to te.tearDown to clean up.
func (te *test) startServer(ts testpb.TestServiceServer) {
	te.testServer = ts
	te.t.Logf("Running test in %s environment...", te.e.name)
	sopts := []grpc.ServerOption{grpc.MaxConcurrentStreams(te.maxStream)}
	if te.maxMsgSize > 0 {
		sopts = append(sopts, grpc.MaxMsgSize(te.maxMsgSize))
	}
	if te.tapHandle != nil {
		sopts = append(sopts, grpc.InTapHandle(te.tapHandle))
	}
	if te.serverCompression {
		sopts = append(sopts,
			grpc.RPCCompressor(grpc.NewGZIPCompressor()),
			grpc.RPCDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	if te.unaryServerInt != nil {
		sopts = append(sopts, grpc.UnaryInterceptor(te.unaryServerInt))
	}
	if te.streamServerInt != nil {
		sopts = append(sopts, grpc.StreamInterceptor(te.streamServerInt))
	}
	if te.unknownHandler != nil {
		sopts = append(sopts, grpc.UnknownServiceHandler(te.unknownHandler))
	}
	la := "localhost:0"
	switch te.e.network {
	case "unix":
		la = "/tmp/testsock" + fmt.Sprintf("%d", time.Now().UnixNano())
		syscall.Unlink(la)
	}
	lis, err := net.Listen(te.e.network, la)
	if err != nil {
		te.t.Fatalf("Failed to listen: %v", err)
	}
	switch te.e.security {
	case "tls":
		creds, err := credentials.NewServerTLSFromFile(tlsDir+"server1.pem", tlsDir+"server1.key")
		if err != nil {
			te.t.Fatalf("Failed to generate credentials %v", err)
		}
		sopts = append(sopts, grpc.Creds(creds))
	case "clientAlwaysFailCred":
		sopts = append(sopts, grpc.Creds(clientAlwaysFailCred{}))
	case "clientTimeoutCreds":
		sopts = append(sopts, grpc.Creds(&clientTimeoutCreds{}))
	}
	s := grpc.NewServer(sopts...)
	te.srv = s
	if te.e.httpHandler {
		internal.TestingUseHandlerImpl(s)
	}
	if te.healthServer != nil {
		healthpb.RegisterHealthServer(s, te.healthServer)
	}
	if te.testServer != nil {
		testpb.RegisterTestServiceServer(s, te.testServer)
	}
	addr := la
	switch te.e.network {
	case "unix":
	default:
		_, port, err := net.SplitHostPort(lis.Addr().String())
		if err != nil {
			te.t.Fatalf("Failed to parse listener address: %v", err)
		}
		addr = "localhost:" + port
	}

	go s.Serve(lis)
	te.srvAddr = addr
}

func (te *test) clientConn() *grpc.ClientConn {
	if te.cc != nil {
		return te.cc
	}
	opts := []grpc.DialOption{
		grpc.WithDialer(te.e.dialer),
		grpc.WithUserAgent(te.userAgent),
	}

	if te.sc != nil {
		opts = append(opts, grpc.WithServiceConfig(te.sc))
	}

	if te.clientCompression {
		opts = append(opts,
			grpc.WithCompressor(grpc.NewGZIPCompressor()),
			grpc.WithDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	if te.unaryClientInt != nil {
		opts = append(opts, grpc.WithUnaryInterceptor(te.unaryClientInt))
	}
	if te.streamClientInt != nil {
		opts = append(opts, grpc.WithStreamInterceptor(te.streamClientInt))
	}
	if te.maxMsgSize > 0 {
		opts = append(opts, grpc.WithMaxMsgSize(te.maxMsgSize))
	}
	switch te.e.security {
	case "tls":
		creds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", "x.test.youtube.com")
		if err != nil {
			te.t.Fatalf("Failed to load credentials: %v", err)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	case "clientAlwaysFailCred":
		opts = append(opts, grpc.WithTransportCredentials(clientAlwaysFailCred{}))
	case "clientTimeoutCreds":
		opts = append(opts, grpc.WithTransportCredentials(&clientTimeoutCreds{}))
	default:
		opts = append(opts, grpc.WithInsecure())
	}
	if te.e.balancer {
		opts = append(opts, grpc.WithBalancer(grpc.RoundRobin(nil)))
	}
	var err error
	te.cc, err = grpc.Dial(te.srvAddr, opts...)
	if err != nil {
		te.t.Fatalf("Dial(%q) = %v", te.srvAddr, err)
	}
	return te.cc
}

func (te *test) declareLogNoise(phrases ...string) {
	te.restoreLogs = declareLogNoise(te.t, phrases...)
}

func (te *test) withServerTester(fn func(st *serverTester)) {
	c, err := te.e.dialer(te.srvAddr, 10*time.Second)
	if err != nil {
		te.t.Fatal(err)
	}
	defer c.Close()
	if te.e.security == "tls" {
		c = tls.Client(c, &tls.Config{
			InsecureSkipVerify: true,
			NextProtos:         []string{http2.NextProtoTLS},
		})
	}
	st := newServerTesterFromConn(te.t, c)
	st.greet()
	fn(st)
}

func TestTimeoutOnDeadServer(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testTimeoutOnDeadServer(t, e)
	}
}

func testTimeoutOnDeadServer(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	te.srv.Stop()
	ctx, _ := context.WithTimeout(context.Background(), time.Millisecond)
	_, err := tc.EmptyCall(ctx, &testpb.Empty{}, grpc.FailFast(false))
	if e.balancer && grpc.Code(err) != codes.DeadlineExceeded {
		// If e.balancer == nil, the ac will stop reconnecting because the dialer returns non-temp error,
		// the error will be an internal error.
		t.Fatalf("TestService/EmptyCall(%v, _) = _, %v, want _, error code: %s", ctx, err, codes.DeadlineExceeded)
	}
	awaitNewConnLogOutput()
}

func TestServerGracefulStopIdempotent(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testServerGracefulStopIdempotent(t, e)
	}
}

func testServerGracefulStopIdempotent(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	for i := 0; i < 3; i++ {
		te.srv.GracefulStop()
	}
}

func TestServerGoAway(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testServerGoAway(t, e)
	}
}

func testServerGoAway(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	// Finish an RPC to make sure the connection is good.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	ch := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(ch)
	}()
	// Loop until the server side GoAway signal is propagated to the client.
	for {
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if _, err := tc.EmptyCall(ctx, &testpb.Empty{}, grpc.FailFast(false)); err == nil {
			continue
		}
		break
	}
	// A new RPC should fail.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.Unavailable && grpc.Code(err) != codes.Internal {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %s or %s", err, codes.Unavailable, codes.Internal)
	}
	<-ch
	awaitNewConnLogOutput()
}

func TestServerGoAwayPendingRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testServerGoAwayPendingRPC(t, e)
	}
}

func testServerGoAwayPendingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := tc.FullDuplexCall(ctx, grpc.FailFast(false))
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	// Finish an RPC to make sure the connection is good.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("%v.EmptyCall(_, _, _) = _, %v, want _, <nil>", tc, err)
	}
	ch := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(ch)
	}()
	// Loop until the server side GoAway signal is propagated to the client.
	for {
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if _, err := tc.EmptyCall(ctx, &testpb.Empty{}, grpc.FailFast(false)); err == nil {
			continue
		}
		break
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(1),
		},
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(100))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	// The existing RPC should be still good to proceed.
	if err := stream.Send(req); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = _, %v, want _, <nil>", stream, err)
	}
	cancel()
	<-ch
	awaitNewConnLogOutput()
}

func TestServerMultipleGoAwayPendingRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testServerMultipleGoAwayPendingRPC(t, e)
	}
}

func testServerMultipleGoAwayPendingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := tc.FullDuplexCall(ctx, grpc.FailFast(false))
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	// Finish an RPC to make sure the connection is good.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("%v.EmptyCall(_, _, _) = _, %v, want _, <nil>", tc, err)
	}
	ch1 := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(ch1)
	}()
	ch2 := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(ch2)
	}()
	// Loop until the server side GoAway signal is propagated to the client.
	for {
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if _, err := tc.EmptyCall(ctx, &testpb.Empty{}, grpc.FailFast(false)); err == nil {
			continue
		}
		break
	}
	select {
	case <-ch1:
		t.Fatal("GracefulStop() terminated early")
	case <-ch2:
		t.Fatal("GracefulStop() terminated early")
	default:
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(1),
		},
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(100))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	// The existing RPC should be still good to proceed.
	if err := stream.Send(req); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = _, %v, want _, <nil>", stream, err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() = %v, want <nil>", stream, err)
	}
	<-ch1
	<-ch2
	cancel()
	awaitNewConnLogOutput()
}

func TestConcurrentClientConnCloseAndServerGoAway(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testConcurrentClientConnCloseAndServerGoAway(t, e)
	}
}

func testConcurrentClientConnCloseAndServerGoAway(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("%v.EmptyCall(_, _, _) = _, %v, want _, <nil>", tc, err)
	}
	ch := make(chan struct{})
	// Close ClientConn and Server concurrently.
	go func() {
		te.srv.GracefulStop()
		close(ch)
	}()
	go func() {
		cc.Close()
	}()
	<-ch
}

func TestConcurrentServerStopAndGoAway(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testConcurrentServerStopAndGoAway(t, e)
	}
}

func testConcurrentServerStopAndGoAway(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	stream, err := tc.FullDuplexCall(context.Background(), grpc.FailFast(false))
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	// Finish an RPC to make sure the connection is good.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		t.Fatalf("%v.EmptyCall(_, _, _) = _, %v, want _, <nil>", tc, err)
	}
	ch := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(ch)
	}()
	// Loop until the server side GoAway signal is propagated to the client.
	for {
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if _, err := tc.EmptyCall(ctx, &testpb.Empty{}, grpc.FailFast(false)); err == nil {
			continue
		}
		break
	}
	// Stop the server and close all the connections.
	te.srv.Stop()
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(1),
		},
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(100))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	if err := stream.Send(req); err == nil {
		if _, err := stream.Recv(); err == nil {
			t.Fatalf("%v.Recv() = _, %v, want _, <nil>", stream, err)
		}
	}
	<-ch
	awaitNewConnLogOutput()
}

func TestClientConnCloseAfterGoAwayWithActiveStream(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testClientConnCloseAfterGoAwayWithActiveStream(t, e)
	}
}

func testClientConnCloseAfterGoAwayWithActiveStream(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	if _, err := tc.FullDuplexCall(context.Background()); err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want _, <nil>", tc, err)
	}
	done := make(chan struct{})
	go func() {
		te.srv.GracefulStop()
		close(done)
	}()
	time.Sleep(time.Second)
	cc.Close()
	timeout := time.NewTimer(time.Second)
	select {
	case <-done:
	case <-timeout.C:
		t.Fatalf("Test timed-out.")
	}
}

func TestFailFast(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testFailFast(t, e)
	}
}

func testFailFast(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	// Stop the server and tear down all the exisiting connections.
	te.srv.Stop()
	// Loop until the server teardown is propagated to the client.
	for {
		_, err := tc.EmptyCall(context.Background(), &testpb.Empty{})
		if grpc.Code(err) == codes.Unavailable {
			break
		}
		fmt.Printf("%v.EmptyCall(_, _) = _, %v", tc, err)
		time.Sleep(10 * time.Millisecond)
	}
	// The client keeps reconnecting and ongoing fail-fast RPCs should fail with code.Unavailable.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.Unavailable {
		t.Fatalf("TestService/EmptyCall(_, _, _) = _, %v, want _, error code: %s", err, codes.Unavailable)
	}
	if _, err := tc.StreamingInputCall(context.Background()); grpc.Code(err) != codes.Unavailable {
		t.Fatalf("TestService/StreamingInputCall(_) = _, %v, want _, error code: %s", err, codes.Unavailable)
	}

	awaitNewConnLogOutput()
}

func TestServiceConfig(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testServiceConfig(t, e)
	}
}

func testServiceConfig(t *testing.T, e env) {
	te := newTest(t, e)
	ch := make(chan grpc.ServiceConfig)
	te.sc = ch
	te.userAgent = testAppUA
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
		"Failed to dial : context canceled; please retry.",
	)
	defer te.tearDown()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		mc := grpc.MethodConfig{
			WaitForReady: true,
			Timeout:      time.Millisecond,
		}
		m := make(map[string]grpc.MethodConfig)
		m["/grpc.testing.TestService/EmptyCall"] = mc
		m["/grpc.testing.TestService/FullDuplexCall"] = mc
		sc := grpc.ServiceConfig{
			Methods: m,
		}
		ch <- sc
	}()
	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	// The following RPCs are expected to become non-fail-fast ones with 1ms deadline.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %s", err, codes.DeadlineExceeded)
	}
	if _, err := tc.FullDuplexCall(context.Background()); grpc.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want %s", err, codes.DeadlineExceeded)
	}
	wg.Wait()
	// Generate a service config update.
	mc := grpc.MethodConfig{
		WaitForReady: false,
	}
	m := make(map[string]grpc.MethodConfig)
	m["/grpc.testing.TestService/EmptyCall"] = mc
	m["/grpc.testing.TestService/FullDuplexCall"] = mc
	sc := grpc.ServiceConfig{
		Methods: m,
	}
	ch <- sc
	// Loop until the new update becomes effective.
	for {
		if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.Unavailable {
			continue
		}
		break
	}
	// The following RPCs are expected to become fail-fast.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.Unavailable {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %s", err, codes.Unavailable)
	}
	if _, err := tc.FullDuplexCall(context.Background()); grpc.Code(err) != codes.Unavailable {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want %s", err, codes.Unavailable)
	}
}

func TestTap(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testTap(t, e)
	}
}

type myTap struct {
	cnt int
}

func (t *myTap) handle(ctx context.Context, info *tap.Info) (context.Context, error) {
	if info != nil {
		if info.FullMethodName == "/grpc.testing.TestService/EmptyCall" {
			t.cnt++
		} else if info.FullMethodName == "/grpc.testing.TestService/UnaryCall" {
			return nil, fmt.Errorf("tap error")
		}
	}
	return ctx, nil
}

func testTap(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	ttap := &myTap{}
	te.tapHandle = ttap.handle
	te.declareLogNoise(
		"transport: http2Client.notifyError got notified that the client transport was broken EOF",
		"grpc: addrConn.transportMonitor exits due to: grpc: the connection is closing",
		"grpc: addrConn.resetTransport failed to create client transport: connection error",
	)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	if ttap.cnt != 1 {
		t.Fatalf("Get the count in ttap %d, want 1", ttap.cnt)
	}

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, 31)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(45),
		Payload:      payload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); grpc.Code(err) != codes.Unavailable {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, %s", err, codes.Unavailable)
	}
}

func healthCheck(d time.Duration, cc *grpc.ClientConn, serviceName string) (*healthpb.HealthCheckResponse, error) {
	ctx, _ := context.WithTimeout(context.Background(), d)
	hc := healthpb.NewHealthClient(cc)
	req := &healthpb.HealthCheckRequest{
		Service: serviceName,
	}
	return hc.Check(ctx, req)
}

func TestHealthCheckOnSuccess(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testHealthCheckOnSuccess(t, e)
	}
}

func testHealthCheckOnSuccess(t *testing.T, e env) {
	te := newTest(t, e)
	hs := health.NewServer()
	hs.SetServingStatus("grpc.health.v1.Health", 1)
	te.healthServer = hs
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	if _, err := healthCheck(1*time.Second, cc, "grpc.health.v1.Health"); err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
}

func TestHealthCheckOnFailure(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testHealthCheckOnFailure(t, e)
	}
}

func testHealthCheckOnFailure(t *testing.T, e env) {
	defer leakCheck(t)()
	te := newTest(t, e)
	te.declareLogNoise(
		"Failed to dial ",
		"grpc: the client connection is closing; please retry",
	)
	hs := health.NewServer()
	hs.SetServingStatus("grpc.health.v1.HealthCheck", 1)
	te.healthServer = hs
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	wantErr := grpc.Errorf(codes.DeadlineExceeded, "context deadline exceeded")
	if _, err := healthCheck(0*time.Second, cc, "grpc.health.v1.Health"); !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, error code %s", err, codes.DeadlineExceeded)
	}
	awaitNewConnLogOutput()
}

func TestHealthCheckOff(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		// TODO(bradfitz): Temporarily skip this env due to #619.
		if e.name == "handler-tls" {
			continue
		}
		testHealthCheckOff(t, e)
	}
}

func testHealthCheckOff(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	want := grpc.Errorf(codes.Unimplemented, "unknown service grpc.health.v1.Health")
	if _, err := healthCheck(1*time.Second, te.clientConn(), ""); !reflect.DeepEqual(err, want) {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, %v", err, want)
	}
}

func TestUnknownHandler(t *testing.T) {
	defer leakCheck(t)()
	// An example unknownHandler that returns a different code and a different method, making sure that we do not
	// expose what methods are implemented to a client that is not authenticated.
	unknownHandler := func(srv interface{}, stream grpc.ServerStream) error {
		return grpc.Errorf(codes.Unauthenticated, "user unauthenticated")
	}
	for _, e := range listTestEnv() {
		// TODO(bradfitz): Temporarily skip this env due to #619.
		if e.name == "handler-tls" {
			continue
		}
		testUnknownHandler(t, e, unknownHandler)
	}
}

func testUnknownHandler(t *testing.T, e env, unknownHandler grpc.StreamHandler) {
	te := newTest(t, e)
	te.unknownHandler = unknownHandler
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	want := grpc.Errorf(codes.Unauthenticated, "user unauthenticated")
	if _, err := healthCheck(1*time.Second, te.clientConn(), ""); !reflect.DeepEqual(err, want) {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, %v", err, want)
	}
}

func TestHealthCheckServingStatus(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testHealthCheckServingStatus(t, e)
	}
}

func testHealthCheckServingStatus(t *testing.T, e env) {
	te := newTest(t, e)
	hs := health.NewServer()
	te.healthServer = hs
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	out, err := healthCheck(1*time.Second, cc, "")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("Got the serving status %v, want SERVING", out.Status)
	}
	wantErr := grpc.Errorf(codes.NotFound, "unknown service")
	if _, err := healthCheck(1*time.Second, cc, "grpc.health.v1.Health"); !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, error code %s", err, codes.NotFound)
	}
	hs.SetServingStatus("grpc.health.v1.Health", healthpb.HealthCheckResponse_SERVING)
	out, err = healthCheck(1*time.Second, cc, "grpc.health.v1.Health")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("Got the serving status %v, want SERVING", out.Status)
	}
	hs.SetServingStatus("grpc.health.v1.Health", healthpb.HealthCheckResponse_NOT_SERVING)
	out, err = healthCheck(1*time.Second, cc, "grpc.health.v1.Health")
	if err != nil {
		t.Fatalf("Health/Check(_, _) = _, %v, want _, <nil>", err)
	}
	if out.Status != healthpb.HealthCheckResponse_NOT_SERVING {
		t.Fatalf("Got the serving status %v, want NOT_SERVING", out.Status)
	}

}

func TestErrorChanNoIO(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testErrorChanNoIO(t, e)
	}
}

func testErrorChanNoIO(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	tc := testpb.NewTestServiceClient(te.clientConn())
	if _, err := tc.FullDuplexCall(context.Background()); err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
}

func TestEmptyUnaryWithUserAgent(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testEmptyUnaryWithUserAgent(t, e)
	}
}

func testEmptyUnaryWithUserAgent(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	var header metadata.MD
	reply, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.Header(&header))
	if err != nil || !proto.Equal(&testpb.Empty{}, reply) {
		t.Fatalf("TestService/EmptyCall(_, _) = %v, %v, want %v, <nil>", reply, err, &testpb.Empty{})
	}
	if v, ok := header["ua"]; !ok || !strings.HasPrefix(v[0], testAppUA) {
		t.Fatalf("header[\"ua\"] = %q, %t, want string with prefix %q, true", v, ok, testAppUA)
	}

	te.srv.Stop()
}

func TestFailedEmptyUnary(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testFailedEmptyUnary(t, e)
	}
}

func testFailedEmptyUnary(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = failAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	wantErr := detailedError
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %v", err, wantErr)
	}
}

func TestLargeUnary(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testLargeUnary(t, e)
	}
}

func testLargeUnary(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const argSize = 271828
	const respSize = 314159

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
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

func TestExceedMsgLimit(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testExceedMsgLimit(t, e)
	}
}

func testExceedMsgLimit(t *testing.T, e env) {
	te := newTest(t, e)
	te.maxMsgSize = 1024
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	argSize := int32(te.maxMsgSize + 1)
	const smallSize = 1

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}
	smallPayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, smallSize)
	if err != nil {
		t.Fatal(err)
	}

	// test on server side for unary RPC
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(smallSize),
		Payload:      payload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || grpc.Code(err) != codes.Internal {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code: %s", err, codes.Internal)
	}
	// test on client side for unary RPC
	req.ResponseSize = proto.Int32(int32(te.maxMsgSize) + 1)
	req.Payload = smallPayload
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || grpc.Code(err) != codes.Internal {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code: %s", err, codes.Internal)
	}

	// test on server side for streaming RPC
	stream, err := tc.FullDuplexCall(te.ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(1),
		},
	}

	spayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(te.maxMsgSize+1))
	if err != nil {
		t.Fatal(err)
	}

	sreq := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            spayload,
	}
	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err == nil || grpc.Code(err) != codes.Internal {
		t.Fatalf("%v.Recv() = _, %v, want _, error code: %s", stream, err, codes.Internal)
	}

	// test on client side for streaming RPC
	stream, err = tc.FullDuplexCall(te.ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	respParam[0].Size = proto.Int32(int32(te.maxMsgSize) + 1)
	sreq.Payload = smallPayload
	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err == nil || grpc.Code(err) != codes.Internal {
		t.Fatalf("%v.Recv() = _, %v, want _, error code: %s", stream, err, codes.Internal)
	}

}

func TestPeerClientSide(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testPeerClientSide(t, e)
	}
}

func testPeerClientSide(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())
	peer := new(peer.Peer)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.Peer(peer), grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	pa := peer.Addr.String()
	if e.network == "unix" {
		if pa != te.srvAddr {
			t.Fatalf("peer.Addr = %v, want %v", pa, te.srvAddr)
		}
		return
	}
	_, pp, err := net.SplitHostPort(pa)
	if err != nil {
		t.Fatalf("Failed to parse address from peer.")
	}
	_, sp, err := net.SplitHostPort(te.srvAddr)
	if err != nil {
		t.Fatalf("Failed to parse address of test server.")
	}
	if pp != sp {
		t.Fatalf("peer.Addr = localhost:%v, want localhost:%v", pp, sp)
	}
}

// TestPeerNegative tests that if call fails setting peer
// doesn't cause a segmentation fault.
// issue#1141 https://github.com/grpc/grpc-go/issues/1141
func TestPeerNegative(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testPeerNegative(t, e)
	}
}

func testPeerNegative(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	peer := new(peer.Peer)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tc.EmptyCall(ctx, &testpb.Empty{}, grpc.Peer(peer))
}

func TestMetadataUnaryRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testMetadataUnaryRPC(t, e)
	}
}

func testMetadataUnaryRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const argSize = 2718
	const respSize = 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	var header, trailer metadata.MD
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Header(&header), grpc.Trailer(&trailer)); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}
	// Ignore optional response headers that Servers may set:
	if header != nil {
		delete(header, "trailer") // RFC 2616 says server SHOULD (but optional) declare trailers
		delete(header, "date")    // the Date header is also optional
		delete(header, "user-agent")
	}
	if !reflect.DeepEqual(header, testMetadata) {
		t.Fatalf("Received header metadata %v, want %v", header, testMetadata)
	}
	if !reflect.DeepEqual(trailer, testTrailerMetadata) {
		t.Fatalf("Received trailer metadata %v, want %v", trailer, testTrailerMetadata)
	}
}

func TestMultipleSetTrailerUnaryRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testMultipleSetTrailerUnaryRPC(t, e)
	}
}

func testMultipleSetTrailerUnaryRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, multipleSetTrailer: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = 1
	)
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	var trailer metadata.MD
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Trailer(&trailer), grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}
	expectedTrailer := metadata.Join(testTrailerMetadata, testTrailerMetadata2)
	if !reflect.DeepEqual(trailer, expectedTrailer) {
		t.Fatalf("Received trailer metadata %v, want %v", trailer, expectedTrailer)
	}
}

func TestMultipleSetTrailerStreamingRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testMultipleSetTrailerStreamingRPC(t, e)
	}
}

func testMultipleSetTrailerStreamingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, multipleSetTrailer: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	stream, err := tc.FullDuplexCall(ctx, grpc.FailFast(false))
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("%v failed to complele the FullDuplexCall: %v", stream, err)
	}

	trailer := stream.Trailer()
	expectedTrailer := metadata.Join(testTrailerMetadata, testTrailerMetadata2)
	if !reflect.DeepEqual(trailer, expectedTrailer) {
		t.Fatalf("Received trailer metadata %v, want %v", trailer, expectedTrailer)
	}
}

func TestSetAndSendHeaderUnaryRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testSetAndSendHeaderUnaryRPC(t, e)
	}
}

// To test header metadata is sent on SendHeader().
func testSetAndSendHeaderUnaryRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setAndSendHeader: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = 1
	)
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	var header metadata.MD
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Header(&header), grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}
}

func TestMultipleSetHeaderUnaryRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testMultipleSetHeaderUnaryRPC(t, e)
	}
}

// To test header metadata is sent when sending response.
func testMultipleSetHeaderUnaryRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setHeaderOnly: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = 1
	)
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}

	var header metadata.MD
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Header(&header), grpc.FailFast(false)); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}
}

func TestMultipleSetHeaderUnaryRPCError(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testMultipleSetHeaderUnaryRPCError(t, e)
	}
}

// To test header metadata is sent when sending status.
func testMultipleSetHeaderUnaryRPCError(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setHeaderOnly: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = -1 // Invalid respSize to make RPC fail.
	)
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	var header metadata.MD
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	if _, err := tc.UnaryCall(ctx, req, grpc.Header(&header), grpc.FailFast(false)); err == nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <non-nil>", ctx, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}
}

func TestSetAndSendHeaderStreamingRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testSetAndSendHeaderStreamingRPC(t, e)
	}
}

// To test header metadata is sent on SendHeader().
func testSetAndSendHeaderStreamingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setAndSendHeader: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = 1
	)
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("%v failed to complele the FullDuplexCall: %v", stream, err)
	}

	header, err := stream.Header()
	if err != nil {
		t.Fatalf("%v.Header() = _, %v, want _, <nil>", stream, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}
}

func TestMultipleSetHeaderStreamingRPC(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testMultipleSetHeaderStreamingRPC(t, e)
	}
}

// To test header metadata is sent when sending response.
func testMultipleSetHeaderStreamingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setHeaderOnly: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = 1
	)
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.StreamingOutputCallRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: []*testpb.ResponseParameters{
			{Size: proto.Int32(respSize)},
		},
		Payload: payload,
	}
	if err := stream.Send(req); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = %v, want <nil>", stream, err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("%v failed to complele the FullDuplexCall: %v", stream, err)
	}

	header, err := stream.Header()
	if err != nil {
		t.Fatalf("%v.Header() = _, %v, want _, <nil>", stream, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}

}

func TestMultipleSetHeaderStreamingRPCError(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testMultipleSetHeaderStreamingRPCError(t, e)
	}
}

// To test header metadata is sent when sending status.
func testMultipleSetHeaderStreamingRPCError(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, setHeaderOnly: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const (
		argSize  = 1
		respSize = -1
	)
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.StreamingOutputCallRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: []*testpb.ResponseParameters{
			{Size: proto.Int32(respSize)},
		},
		Payload: payload,
	}
	if err := stream.Send(req); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
	}
	if _, err := stream.Recv(); err == nil {
		t.Fatalf("%v.Recv() = %v, want <non-nil>", stream, err)
	}

	header, err := stream.Header()
	if err != nil {
		t.Fatalf("%v.Header() = _, %v, want _, <nil>", stream, err)
	}
	delete(header, "user-agent")
	expectedHeader := metadata.Join(testMetadata, testMetadata2)
	if !reflect.DeepEqual(header, expectedHeader) {
		t.Fatalf("Received header metadata %v, want %v", header, expectedHeader)
	}

	if err := stream.CloseSend(); err != nil {
		t.Fatalf("%v.CloseSend() got %v, want %v", stream, err, nil)
	}
}

// TestMalformedHTTP2Metedata verfies the returned error when the client
// sends an illegal metadata.
func TestMalformedHTTP2Metadata(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testMalformedHTTP2Metadata(t, e)
	}
}

func testMalformedHTTP2Metadata(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, 2718)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(314),
		Payload:      payload,
	}
	ctx := metadata.NewOutgoingContext(context.Background(), malformedHTTP2Metadata)
	if _, err := tc.UnaryCall(ctx, req); grpc.Code(err) != codes.Internal {
		t.Fatalf("TestService.UnaryCall(%v, _) = _, %v; want _, %s", ctx, err, codes.Internal)
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
	reply, err := tc.UnaryCall(context.Background(), req, grpc.FailFast(false))
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
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testRetry(t, e)
	}
}

// This test mimics a user who sends 1000 RPCs concurrently on a faulty transport.
// TODO(zhaoq): Refactor to make this clearer and add more cases to test racy
// and error-prone paths.
func testRetry(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise("transport: http2Client.notifyError got notified that the client transport was broken")
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	var wg sync.WaitGroup

	numRPC := 1000
	rpcSpacing := 2 * time.Millisecond
	if raceMode {
		// The race detector has a limit on how many goroutines it can track.
		// This test is near the upper limit, and goes over the limit
		// depending on the environment (the http.Handler environment uses
		// more goroutines)
		t.Logf("Shortening test in race mode.")
		numRPC /= 2
		rpcSpacing *= 2
	}

	wg.Add(1)
	go func() {
		// Halfway through starting RPCs, kill all connections:
		time.Sleep(time.Duration(numRPC/2) * rpcSpacing)

		// The server shuts down the network connection to make a
		// transport error which will be detected by the client side
		// code.
		internal.TestingCloseConns(te.srv)
		wg.Done()
	}()
	// All these RPCs should succeed eventually.
	for i := 0; i < numRPC; i++ {
		time.Sleep(rpcSpacing)
		wg.Add(1)
		go performOneRPC(t, tc, &wg)
	}
	wg.Wait()
}

func TestRPCTimeout(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testRPCTimeout(t, e)
	}
}

// TODO(zhaoq): Have a better test coverage of timeout and cancellation mechanism.
func testRPCTimeout(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	const argSize = 2718
	const respSize = 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	for i := -1; i <= 10; i++ {
		ctx, _ := context.WithTimeout(context.Background(), time.Duration(i)*time.Millisecond)
		if _, err := tc.UnaryCall(ctx, req); grpc.Code(err) != codes.DeadlineExceeded {
			t.Fatalf("TestService/UnaryCallv(_, _) = _, %v; want <nil>, error code: %s", err, codes.DeadlineExceeded)
		}
	}
}

func TestCancel(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testCancel(t, e)
	}
}

func testCancel(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise("grpc: the client connection is closing; please retry")
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	const argSize = 2718
	const respSize = 314

	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(1*time.Millisecond, cancel)
	if r, err := tc.UnaryCall(ctx, req); grpc.Code(err) != codes.Canceled {
		t.Fatalf("TestService/UnaryCall(_, _) = %v, %v; want _, error code: %s", r, err, codes.Canceled)
	}
	awaitNewConnLogOutput()
}

func TestCancelNoIO(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testCancelNoIO(t, e)
	}
}

func testCancelNoIO(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise("http2Client.notifyError got notified that the client transport was broken")
	te.maxStream = 1 // Only allows 1 live stream per server transport.
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	// Start one blocked RPC for which we'll never send streaming
	// input. This will consume the 1 maximum concurrent streams,
	// causing future RPCs to hang.
	ctx, cancelFirst := context.WithCancel(context.Background())
	_, err := tc.StreamingInputCall(ctx)
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
	}

	// Loop until the ClientConn receives the initial settings
	// frame from the server, notifying it about the maximum
	// concurrent streams. We know when it's received it because
	// an RPC will fail with codes.DeadlineExceeded instead of
	// succeeding.
	// TODO(bradfitz): add internal test hook for this (Issue 534)
	for {
		ctx, cancelSecond := context.WithTimeout(context.Background(), 250*time.Millisecond)
		_, err := tc.StreamingInputCall(ctx)
		cancelSecond()
		if err == nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if grpc.Code(err) == codes.DeadlineExceeded {
			break
		}
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %s", tc, err, codes.DeadlineExceeded)
	}
	// If there are any RPCs in flight before the client receives
	// the max streams setting, let them be expired.
	// TODO(bradfitz): add internal test hook for this (Issue 534)
	time.Sleep(500 * time.Millisecond)

	ch := make(chan struct{})
	go func() {
		defer close(ch)

		// This should be blocked until the 1st is canceled.
		ctx, cancelThird := context.WithTimeout(context.Background(), 2*time.Second)
		if _, err := tc.StreamingInputCall(ctx); err != nil {
			t.Errorf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
		}
		cancelThird()
	}()
	cancelFirst()
	<-ch
}

// The following tests the gRPC streaming RPC implementations.
// TODO(zhaoq): Have better coverage on error cases.
var (
	reqSizes  = []int{27182, 8, 1828, 45904}
	respSizes = []int{31415, 9, 2653, 58979}
)

func TestNoService(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testNoService(t, e)
	}
}

func testNoService(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(nil)
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	stream, err := tc.FullDuplexCall(te.ctx, grpc.FailFast(false))
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	if _, err := stream.Recv(); grpc.Code(err) != codes.Unimplemented {
		t.Fatalf("stream.Recv() = _, %v, want _, error code %s", err, codes.Unimplemented)
	}
}

func TestPingPong(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testPingPong(t, e)
	}
}

func testPingPong(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	stream, err := tc.FullDuplexCall(te.ctx)
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
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testMetadataStreamingRPC(t, e)
	}
}

func testMetadataStreamingRPC(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	ctx := metadata.NewOutgoingContext(te.ctx, testMetadata)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	go func() {
		headerMD, err := stream.Header()
		if e.security == "tls" {
			delete(headerMD, "transport_security_type")
		}
		delete(headerMD, "trailer") // ignore if present
		delete(headerMD, "user-agent")
		if err != nil || !reflect.DeepEqual(testMetadata, headerMD) {
			t.Errorf("#1 %v.Header() = %v, %v, want %v, <nil>", stream, headerMD, err, testMetadata)
		}
		// test the cached value.
		headerMD, err = stream.Header()
		delete(headerMD, "trailer") // ignore if present
		delete(headerMD, "user-agent")
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
	if !reflect.DeepEqual(testTrailerMetadata, trailerMD) {
		t.Fatalf("%v.Trailer() = %v, want %v", stream, trailerMD, testTrailerMetadata)
	}
}

func TestServerStreaming(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testServerStreaming(t, e)
	}
}

func testServerStreaming(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

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
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testFailedServerStreaming(t, e)
	}
}

func testFailedServerStreaming(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = failAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

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
	ctx := metadata.NewOutgoingContext(te.ctx, testMetadata)
	stream, err := tc.StreamingOutputCall(ctx, req)
	if err != nil {
		t.Fatalf("%v.StreamingOutputCall(_) = _, %v, want <nil>", tc, err)
	}
	wantErr := grpc.Errorf(codes.DataLoss, "error for testing: "+failAppUA)
	if _, err := stream.Recv(); !reflect.DeepEqual(err, wantErr) {
		t.Fatalf("%v.Recv() = _, %v, want _, %v", stream, err, wantErr)
	}
}

// concurrentSendServer is a TestServiceServer whose
// StreamingOutputCall makes ten serial Send calls, sending payloads
// "0".."9", inclusive.  TestServerStreamingConcurrent verifies they
// were received in the correct order, and that there were no races.
//
// All other TestServiceServer methods crash if called.
type concurrentSendServer struct {
	testpb.TestServiceServer
}

func (s concurrentSendServer) StreamingOutputCall(args *testpb.StreamingOutputCallRequest, stream testpb.TestService_StreamingOutputCallServer) error {
	for i := 0; i < 10; i++ {
		stream.Send(&testpb.StreamingOutputCallResponse{
			Payload: &testpb.Payload{
				Body: []byte{'0' + uint8(i)},
			},
		})
	}
	return nil
}

// Tests doing a bunch of concurrent streaming output calls.
func TestServerStreamingConcurrent(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testServerStreamingConcurrent(t, e)
	}
}

func testServerStreamingConcurrent(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(concurrentSendServer{})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	doStreamingCall := func() {
		req := &testpb.StreamingOutputCallRequest{}
		stream, err := tc.StreamingOutputCall(context.Background(), req)
		if err != nil {
			t.Errorf("%v.StreamingOutputCall(_) = _, %v, want <nil>", tc, err)
			return
		}
		var ngot int
		var buf bytes.Buffer
		for {
			reply, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			ngot++
			if buf.Len() > 0 {
				buf.WriteByte(',')
			}
			buf.Write(reply.GetPayload().GetBody())
		}
		if want := 10; ngot != want {
			t.Errorf("Got %d replies, want %d", ngot, want)
		}
		if got, want := buf.String(), "0,1,2,3,4,5,6,7,8,9"; got != want {
			t.Errorf("Got replies %q; want %q", got, want)
		}
	}

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			doStreamingCall()
		}()
	}
	wg.Wait()

}

func TestClientStreaming(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testClientStreaming(t, e)
	}
}

func testClientStreaming(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	stream, err := tc.StreamingInputCall(te.ctx)
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

func TestClientStreamingError(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		if e.name == "handler-tls" {
			continue
		}
		testClientStreamingError(t, e)
	}
}

func testClientStreamingError(t *testing.T, e env) {
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security, earlyFail: true})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	stream, err := tc.StreamingInputCall(te.ctx)
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want <nil>", tc, err)
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, 1)
	if err != nil {
		t.Fatal(err)
	}

	req := &testpb.StreamingInputCallRequest{
		Payload: payload,
	}
	// The 1st request should go through.
	if err := stream.Send(req); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, req, err)
	}
	for {
		if err := stream.Send(req); err != io.EOF {
			continue
		}
		if _, err := stream.CloseAndRecv(); grpc.Code(err) != codes.NotFound {
			t.Fatalf("%v.CloseAndRecv() = %v, want error %s", stream, err, codes.NotFound)
		}
		break
	}
}

func TestExceedMaxStreamsLimit(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testExceedMaxStreamsLimit(t, e)
	}
}

func testExceedMaxStreamsLimit(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise(
		"http2Client.notifyError got notified that the client transport was broken",
		"Conn.resetTransport failed to create client transport",
		"grpc: the connection is closing",
	)
	te.maxStream = 1 // Only allows 1 live stream per server transport.
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	_, err := tc.StreamingInputCall(te.ctx)
	if err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
	}
	// Loop until receiving the new max stream setting from the server.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_, err := tc.StreamingInputCall(ctx)
		if err == nil {
			time.Sleep(time.Second)
			continue
		}
		if grpc.Code(err) == codes.DeadlineExceeded {
			break
		}
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %s", tc, err, codes.DeadlineExceeded)
	}
}

const defaultMaxStreamsClient = 100

func TestExceedDefaultMaxStreamsLimit(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testExceedDefaultMaxStreamsLimit(t, e)
	}
}

func testExceedDefaultMaxStreamsLimit(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise(
		"http2Client.notifyError got notified that the client transport was broken",
		"Conn.resetTransport failed to create client transport",
		"grpc: the connection is closing",
	)
	// When masStream is set to 0 the server doesn't send a settings frame for
	// MaxConcurrentStreams, essentially allowing infinite (math.MaxInt32) streams.
	// In such a case, there should be a default cap on the client-side.
	te.maxStream = 0
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	// Create as many streams as a client can.
	for i := 0; i < defaultMaxStreamsClient; i++ {
		if _, err := tc.StreamingInputCall(te.ctx); err != nil {
			t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
		}
	}

	// Trying to create one more should timeout.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := tc.StreamingInputCall(ctx)
	if err == nil || grpc.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %s", tc, err, codes.DeadlineExceeded)
	}
}

func TestStreamsQuotaRecovery(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testStreamsQuotaRecovery(t, e)
	}
}

func testStreamsQuotaRecovery(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise(
		"http2Client.notifyError got notified that the client transport was broken",
		"Conn.resetTransport failed to create client transport",
		"grpc: the connection is closing",
	)
	te.maxStream = 1 // Allows 1 live stream.
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.StreamingInputCall(context.Background()); err != nil {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, <nil>", tc, err)
	}
	// Loop until the new max stream setting is effective.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_, err := tc.StreamingInputCall(ctx)
		if err == nil {
			time.Sleep(time.Second)
			continue
		}
		if grpc.Code(err) == codes.DeadlineExceeded {
			break
		}
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, %s", tc, err, codes.DeadlineExceeded)
	}

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, 314)
			if err != nil {
				t.Fatal(err)
			}
			req := &testpb.SimpleRequest{
				ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
				ResponseSize: proto.Int32(1592),
				Payload:      payload,
			}
			// No rpc should go through due to the max streams limit.
			ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
			if _, err := tc.UnaryCall(ctx, req, grpc.FailFast(false)); grpc.Code(err) != codes.DeadlineExceeded {
				t.Errorf("TestService/UnaryCall(_, _) = _, %v, want _, %s", err, codes.DeadlineExceeded)
			}
		}()
	}
	wg.Wait()
}

func TestCompressServerHasNoSupport(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testCompressServerHasNoSupport(t, e)
	}
}

func testCompressServerHasNoSupport(t *testing.T, e env) {
	te := newTest(t, e)
	te.serverCompression = false
	te.clientCompression = true
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	const argSize = 271828
	const respSize = 314159
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || grpc.Code(err) != codes.Unimplemented {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code %s", err, codes.Unimplemented)
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
	if _, err := stream.Recv(); err == nil || grpc.Code(err) != codes.Unimplemented {
		t.Fatalf("%v.Recv() = %v, want error code %s", stream, err, codes.Unimplemented)
	}
}

func TestCompressOK(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testCompressOK(t, e)
	}
}

func testCompressOK(t *testing.T, e env) {
	te := newTest(t, e)
	te.serverCompression = true
	te.clientCompression = true
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(te.clientConn())

	// Unary call
	const argSize = 271828
	const respSize = 314159
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, argSize)
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseSize: proto.Int32(respSize),
		Payload:      payload,
	}
	ctx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs("something", "something"))
	if _, err := tc.UnaryCall(ctx, req); err != nil {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
	}
	// Streaming RPC
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := tc.FullDuplexCall(ctx)
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

func TestUnaryClientInterceptor(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testUnaryClientInterceptor(t, e)
	}
}

func failOkayRPC(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	err := invoker(ctx, method, req, reply, cc, opts...)
	if err == nil {
		return grpc.Errorf(codes.NotFound, "")
	}
	return err
}

func testUnaryClientInterceptor(t *testing.T, e env) {
	te := newTest(t, e)
	te.userAgent = testAppUA
	te.unaryClientInt = failOkayRPC
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	tc := testpb.NewTestServiceClient(te.clientConn())
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.NotFound {
		t.Fatalf("%v.EmptyCall(_, _) = _, %v, want _, error code %s", tc, err, codes.NotFound)
	}
}

func TestStreamClientInterceptor(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testStreamClientInterceptor(t, e)
	}
}

func failOkayStream(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	s, err := streamer(ctx, desc, cc, method, opts...)
	if err == nil {
		return nil, grpc.Errorf(codes.NotFound, "")
	}
	return s, nil
}

func testStreamClientInterceptor(t *testing.T, e env) {
	te := newTest(t, e)
	te.streamClientInt = failOkayStream
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	tc := testpb.NewTestServiceClient(te.clientConn())
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(int32(1)),
		},
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(1))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	if _, err := tc.StreamingOutputCall(context.Background(), req); grpc.Code(err) != codes.NotFound {
		t.Fatalf("%v.StreamingOutputCall(_) = _, %v, want _, error code %s", tc, err, codes.NotFound)
	}
}

func TestUnaryServerInterceptor(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testUnaryServerInterceptor(t, e)
	}
}

func errInjector(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	return nil, grpc.Errorf(codes.PermissionDenied, "")
}

func testUnaryServerInterceptor(t *testing.T, e env) {
	te := newTest(t, e)
	te.unaryServerInt = errInjector
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	tc := testpb.NewTestServiceClient(te.clientConn())
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); grpc.Code(err) != codes.PermissionDenied {
		t.Fatalf("%v.EmptyCall(_, _) = _, %v, want _, error code %s", tc, err, codes.PermissionDenied)
	}
}

func TestStreamServerInterceptor(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		// TODO(bradfitz): Temporarily skip this env due to #619.
		if e.name == "handler-tls" {
			continue
		}
		testStreamServerInterceptor(t, e)
	}
}

func fullDuplexOnly(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	if info.FullMethod == "/grpc.testing.TestService/FullDuplexCall" {
		return handler(srv, ss)
	}
	// Reject the other methods.
	return grpc.Errorf(codes.PermissionDenied, "")
}

func testStreamServerInterceptor(t *testing.T, e env) {
	te := newTest(t, e)
	te.streamServerInt = fullDuplexOnly
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	tc := testpb.NewTestServiceClient(te.clientConn())
	respParam := []*testpb.ResponseParameters{
		{
			Size: proto.Int32(int32(1)),
		},
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, int32(1))
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.StreamingOutputCallRequest{
		ResponseType:       testpb.PayloadType_COMPRESSABLE.Enum(),
		ResponseParameters: respParam,
		Payload:            payload,
	}
	s1, err := tc.StreamingOutputCall(context.Background(), req)
	if err != nil {
		t.Fatalf("%v.StreamingOutputCall(_) = _, %v, want _, <nil>", tc, err)
	}
	if _, err := s1.Recv(); grpc.Code(err) != codes.PermissionDenied {
		t.Fatalf("%v.StreamingInputCall(_) = _, %v, want _, error code %s", tc, err, codes.PermissionDenied)
	}
	s2, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	if err := s2.Send(req); err != nil {
		t.Fatalf("%v.Send(_) = %v, want <nil>", s2, err)
	}
	if _, err := s2.Recv(); err != nil {
		t.Fatalf("%v.Recv() = _, %v, want _, <nil>", s2, err)
	}
}

// funcServer implements methods of TestServiceServer using funcs,
// similar to an http.HandlerFunc.
// Any unimplemented method will crash. Tests implement the method(s)
// they need.
type funcServer struct {
	testpb.TestServiceServer
	unaryCall          func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error)
	streamingInputCall func(stream testpb.TestService_StreamingInputCallServer) error
}

func (s *funcServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	return s.unaryCall(ctx, in)
}

func (s *funcServer) StreamingInputCall(stream testpb.TestService_StreamingInputCallServer) error {
	return s.streamingInputCall(stream)
}

func TestClientRequestBodyErrorUnexpectedEOF(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testClientRequestBodyErrorUnexpectedEOF(t, e)
	}
}

func testClientRequestBodyErrorUnexpectedEOF(t *testing.T, e env) {
	te := newTest(t, e)
	ts := &funcServer{unaryCall: func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
		errUnexpectedCall := errors.New("unexpected call func server method")
		t.Error(errUnexpectedCall)
		return nil, errUnexpectedCall
	}}
	te.startServer(ts)
	defer te.tearDown()
	te.withServerTester(func(st *serverTester) {
		st.writeHeadersGRPC(1, "/grpc.testing.TestService/UnaryCall")
		// Say we have 5 bytes coming, but set END_STREAM flag:
		st.writeData(1, true, []byte{0, 0, 0, 0, 5})
		st.wantAnyFrame() // wait for server to crash (it used to crash)
	})
}

func TestClientRequestBodyErrorCloseAfterLength(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testClientRequestBodyErrorCloseAfterLength(t, e)
	}
}

func testClientRequestBodyErrorCloseAfterLength(t *testing.T, e env) {
	te := newTest(t, e)
	te.declareLogNoise("Server.processUnaryRPC failed to write status")
	ts := &funcServer{unaryCall: func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
		errUnexpectedCall := errors.New("unexpected call func server method")
		t.Error(errUnexpectedCall)
		return nil, errUnexpectedCall
	}}
	te.startServer(ts)
	defer te.tearDown()
	te.withServerTester(func(st *serverTester) {
		st.writeHeadersGRPC(1, "/grpc.testing.TestService/UnaryCall")
		// say we're sending 5 bytes, but then close the connection instead.
		st.writeData(1, false, []byte{0, 0, 0, 0, 5})
		st.cc.Close()
	})
}

func TestClientRequestBodyErrorCancel(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testClientRequestBodyErrorCancel(t, e)
	}
}

func testClientRequestBodyErrorCancel(t *testing.T, e env) {
	te := newTest(t, e)
	gotCall := make(chan bool, 1)
	ts := &funcServer{unaryCall: func(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
		gotCall <- true
		return new(testpb.SimpleResponse), nil
	}}
	te.startServer(ts)
	defer te.tearDown()
	te.withServerTester(func(st *serverTester) {
		st.writeHeadersGRPC(1, "/grpc.testing.TestService/UnaryCall")
		// Say we have 5 bytes coming, but cancel it instead.
		st.writeRSTStream(1, http2.ErrCodeCancel)
		st.writeData(1, false, []byte{0, 0, 0, 0, 5})

		// Verify we didn't a call yet.
		select {
		case <-gotCall:
			t.Fatal("unexpected call")
		default:
		}

		// And now send an uncanceled (but still invalid), just to get a response.
		st.writeHeadersGRPC(3, "/grpc.testing.TestService/UnaryCall")
		st.writeData(3, true, []byte{0, 0, 0, 0, 0})
		<-gotCall
		st.wantAnyFrame()
	})
}

func TestClientRequestBodyErrorCancelStreamingInput(t *testing.T) {
	defer leakCheck(t)()
	for _, e := range listTestEnv() {
		testClientRequestBodyErrorCancelStreamingInput(t, e)
	}
}

func testClientRequestBodyErrorCancelStreamingInput(t *testing.T, e env) {
	te := newTest(t, e)
	recvErr := make(chan error, 1)
	ts := &funcServer{streamingInputCall: func(stream testpb.TestService_StreamingInputCallServer) error {
		_, err := stream.Recv()
		recvErr <- err
		return nil
	}}
	te.startServer(ts)
	defer te.tearDown()
	te.withServerTester(func(st *serverTester) {
		st.writeHeadersGRPC(1, "/grpc.testing.TestService/StreamingInputCall")
		// Say we have 5 bytes coming, but cancel it instead.
		st.writeData(1, false, []byte{0, 0, 0, 0, 5})
		st.writeRSTStream(1, http2.ErrCodeCancel)

		var got error
		select {
		case got = <-recvErr:
		case <-time.After(3 * time.Second):
			t.Fatal("timeout waiting for error")
		}
		if grpc.Code(got) != codes.Canceled {
			t.Errorf("error = %#v; want error code %s", got, codes.Canceled)
		}
	})
}

const clientAlwaysFailCredErrorMsg = "clientAlwaysFailCred always fails"

var errClientAlwaysFailCred = errors.New(clientAlwaysFailCredErrorMsg)

type clientAlwaysFailCred struct{}

func (c clientAlwaysFailCred) ClientHandshake(ctx context.Context, addr string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errClientAlwaysFailCred
}
func (c clientAlwaysFailCred) ServerHandshake(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return rawConn, nil, nil
}
func (c clientAlwaysFailCred) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{}
}
func (c clientAlwaysFailCred) Clone() credentials.TransportCredentials {
	return nil
}
func (c clientAlwaysFailCred) OverrideServerName(s string) error {
	return nil
}

func TestDialWithBlockErrorOnBadCertificates(t *testing.T) {
	te := newTest(t, env{name: "bad-cred", network: "tcp", security: "clientAlwaysFailCred", balancer: true})
	te.startServer(&testServer{security: te.e.security})
	defer te.tearDown()

	var (
		err  error
		opts []grpc.DialOption
	)
	opts = append(opts, grpc.WithTransportCredentials(clientAlwaysFailCred{}), grpc.WithBlock())
	te.cc, err = grpc.Dial(te.srvAddr, opts...)
	if err != errClientAlwaysFailCred {
		te.t.Fatalf("Dial(%q) = %v, want %v", te.srvAddr, err, errClientAlwaysFailCred)
	}
}

func TestFailFastRPCErrorOnBadCertificates(t *testing.T) {
	te := newTest(t, env{name: "bad-cred", network: "tcp", security: "clientAlwaysFailCred", balancer: true})
	te.startServer(&testServer{security: te.e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); !strings.Contains(err.Error(), clientAlwaysFailCredErrorMsg) {
		te.t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want err.Error() contains %q", err, clientAlwaysFailCredErrorMsg)
	}
}

func TestFailFastRPCWithNoBalancerErrorOnBadCertificates(t *testing.T) {
	te := newTest(t, env{name: "bad-cred", network: "tcp", security: "clientAlwaysFailCred", balancer: false})
	te.startServer(&testServer{security: te.e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); !strings.Contains(err.Error(), clientAlwaysFailCredErrorMsg) {
		te.t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want err.Error() contains %q", err, clientAlwaysFailCredErrorMsg)
	}
}

func TestNonFailFastRPCWithNoBalancerErrorOnBadCertificates(t *testing.T) {
	te := newTest(t, env{name: "bad-cred", network: "tcp", security: "clientAlwaysFailCred", balancer: false})
	te.startServer(&testServer{security: te.e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); !strings.Contains(err.Error(), clientAlwaysFailCredErrorMsg) {
		te.t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want err.Error() contains %q", err, clientAlwaysFailCredErrorMsg)
	}
}

type clientTimeoutCreds struct {
	timeoutReturned bool
}

func (c *clientTimeoutCreds) ClientHandshake(ctx context.Context, addr string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	if !c.timeoutReturned {
		c.timeoutReturned = true
		return nil, nil, context.DeadlineExceeded
	}
	return rawConn, nil, nil
}
func (c *clientTimeoutCreds) ServerHandshake(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return rawConn, nil, nil
}
func (c *clientTimeoutCreds) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{}
}
func (c *clientTimeoutCreds) Clone() credentials.TransportCredentials {
	return nil
}
func (c *clientTimeoutCreds) OverrideServerName(s string) error {
	return nil
}

func TestNonFailFastRPCSucceedOnTimeoutCreds(t *testing.T) {
	te := newTest(t, env{name: "timeout-cred", network: "tcp", security: "clientTimeoutCreds", balancer: false})
	te.userAgent = testAppUA
	te.startServer(&testServer{security: te.e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	// This unary call should succeed, because ClientHandshake will succeed for the second time.
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}, grpc.FailFast(false)); err != nil {
		te.t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want <nil>", err)
	}
}

type serverDispatchCred struct {
	ready   chan struct{}
	rawConn net.Conn
}

func newServerDispatchCred() *serverDispatchCred {
	return &serverDispatchCred{
		ready: make(chan struct{}),
	}
}
func (c *serverDispatchCred) ClientHandshake(ctx context.Context, addr string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return rawConn, nil, nil
}
func (c *serverDispatchCred) ServerHandshake(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	c.rawConn = rawConn
	close(c.ready)
	return nil, nil, credentials.ErrConnDispatched
}
func (c *serverDispatchCred) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{}
}
func (c *serverDispatchCred) Clone() credentials.TransportCredentials {
	return nil
}
func (c *serverDispatchCred) OverrideServerName(s string) error {
	return nil
}
func (c *serverDispatchCred) getRawConn() net.Conn {
	<-c.ready
	return c.rawConn
}

func TestServerCredsDispatch(t *testing.T) {
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	cred := newServerDispatchCred()
	s := grpc.NewServer(grpc.Creds(cred))
	go s.Serve(lis)
	defer s.Stop()

	cc, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(cred))
	if err != nil {
		t.Fatalf("grpc.Dial(%q) = %v", lis.Addr().String(), err)
	}
	defer cc.Close()

	// Check rawConn is not closed.
	if n, err := cred.getRawConn().Write([]byte{0}); n <= 0 || err != nil {
		t.Errorf("Read() = %v, %v; want n>0, <nil>", n, err)
	}
}

func TestFlowControlLogicalRace(t *testing.T) {
	// Test for a regression of https://github.com/grpc/grpc-go/issues/632,
	// and other flow control bugs.

	defer leakCheck(t)()

	const (
		itemCount   = 100
		itemSize    = 1 << 10
		recvCount   = 2
		maxFailures = 3

		requestTimeout = time.Second
	)

	requestCount := 10000
	if raceMode {
		requestCount = 1000
	}

	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	defer lis.Close()

	s := grpc.NewServer()
	testpb.RegisterTestServiceServer(s, &flowControlLogicalRaceServer{
		itemCount: itemCount,
		itemSize:  itemSize,
	})
	defer s.Stop()

	go s.Serve(lis)

	ctx := context.Background()

	cc, err := grpc.Dial(lis.Addr().String(), grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		t.Fatalf("grpc.Dial(%q) = %v", lis.Addr().String(), err)
	}
	defer cc.Close()
	cl := testpb.NewTestServiceClient(cc)

	failures := 0
	for i := 0; i < requestCount; i++ {
		ctx, cancel := context.WithTimeout(ctx, requestTimeout)
		output, err := cl.StreamingOutputCall(ctx, &testpb.StreamingOutputCallRequest{})
		if err != nil {
			t.Fatalf("StreamingOutputCall; err = %q", err)
		}

		j := 0
	loop:
		for ; j < recvCount; j++ {
			_, err := output.Recv()
			if err != nil {
				if err == io.EOF {
					break loop
				}
				switch grpc.Code(err) {
				case codes.DeadlineExceeded:
					break loop
				default:
					t.Fatalf("Recv; err = %q", err)
				}
			}
		}
		cancel()
		<-ctx.Done()

		if j < recvCount {
			t.Errorf("got %d responses to request %d", j, i)
			failures++
			if failures >= maxFailures {
				// Continue past the first failure to see if the connection is
				// entirely broken, or if only a single RPC was affected
				break
			}
		}
	}
}

type flowControlLogicalRaceServer struct {
	testpb.TestServiceServer

	itemSize  int
	itemCount int
}

func (s *flowControlLogicalRaceServer) StreamingOutputCall(req *testpb.StreamingOutputCallRequest, srv testpb.TestService_StreamingOutputCallServer) error {
	for i := 0; i < s.itemCount; i++ {
		err := srv.Send(&testpb.StreamingOutputCallResponse{
			Payload: &testpb.Payload{
				// Sending a large stream of data which the client reject
				// helps to trigger some types of flow control bugs.
				//
				// Reallocating memory here is inefficient, but the stress it
				// puts on the GC leads to more frequent flow control
				// failures. The GC likely causes more variety in the
				// goroutine scheduling orders.
				Body: bytes.Repeat([]byte("a"), s.itemSize),
			},
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// interestingGoroutines returns all goroutines we care about for the purpose
// of leak checking. It excludes testing or runtime ones.
func interestingGoroutines() (gs []string) {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	for _, g := range strings.Split(string(buf), "\n\n") {
		sl := strings.SplitN(g, "\n", 2)
		if len(sl) != 2 {
			continue
		}
		stack := strings.TrimSpace(sl[1])
		if strings.HasPrefix(stack, "testing.RunTests") {
			continue
		}

		if stack == "" ||
			strings.Contains(stack, "testing.Main(") ||
			strings.Contains(stack, "testing.tRunner(") ||
			strings.Contains(stack, "runtime.goexit") ||
			strings.Contains(stack, "created by runtime.gc") ||
			strings.Contains(stack, "created by runtime/trace.Start") ||
			strings.Contains(stack, "created by google3/base/go/log.init") ||
			strings.Contains(stack, "interestingGoroutines") ||
			strings.Contains(stack, "runtime.MHeap_Scavenger") ||
			strings.Contains(stack, "signal.signal_recv") ||
			strings.Contains(stack, "sigterm.handler") ||
			strings.Contains(stack, "runtime_mcall") ||
			strings.Contains(stack, "goroutine in C code") {
			continue
		}
		gs = append(gs, g)
	}
	sort.Strings(gs)
	return
}

// leakCheck snapshots the currently-running goroutines and returns a
// function to be run at the end of tests to see whether any
// goroutines leaked.
func leakCheck(t testing.TB) func() {
	orig := map[string]bool{}
	for _, g := range interestingGoroutines() {
		orig[g] = true
	}
	return func() {
		// Loop, waiting for goroutines to shut down.
		// Wait up to 10 seconds, but finish as quickly as possible.
		deadline := time.Now().Add(10 * time.Second)
		for {
			var leaked []string
			for _, g := range interestingGoroutines() {
				if !orig[g] {
					leaked = append(leaked, g)
				}
			}
			if len(leaked) == 0 {
				return
			}
			if time.Now().Before(deadline) {
				time.Sleep(50 * time.Millisecond)
				continue
			}
			for _, g := range leaked {
				t.Errorf("Leaked goroutine: %v", g)
			}
			return
		}
	}
}

type lockingWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (lw *lockingWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()
	return lw.w.Write(p)
}

func (lw *lockingWriter) setWriter(w io.Writer) {
	lw.mu.Lock()
	defer lw.mu.Unlock()
	lw.w = w
}

var testLogOutput = &lockingWriter{w: os.Stderr}

// awaitNewConnLogOutput waits for any of grpc.NewConn's goroutines to
// terminate, if they're still running. It spams logs with this
// message.  We wait for it so our log filter is still
// active. Otherwise the "defer restore()" at the top of various test
// functions restores our log filter and then the goroutine spams.
func awaitNewConnLogOutput() {
	awaitLogOutput(50*time.Millisecond, "grpc: the client connection is closing; please retry")
}

func awaitLogOutput(maxWait time.Duration, phrase string) {
	pb := []byte(phrase)

	timer := time.NewTimer(maxWait)
	defer timer.Stop()
	wakeup := make(chan bool, 1)
	for {
		if logOutputHasContents(pb, wakeup) {
			return
		}
		select {
		case <-timer.C:
			// Too slow. Oh well.
			return
		case <-wakeup:
		}
	}
}

func logOutputHasContents(v []byte, wakeup chan<- bool) bool {
	testLogOutput.mu.Lock()
	defer testLogOutput.mu.Unlock()
	fw, ok := testLogOutput.w.(*filterWriter)
	if !ok {
		return false
	}
	fw.mu.Lock()
	defer fw.mu.Unlock()
	if bytes.Contains(fw.buf.Bytes(), v) {
		return true
	}
	fw.wakeup = wakeup
	return false
}

func init() {
	grpclog.SetLogger(log.New(testLogOutput, "", log.LstdFlags))
}

var verboseLogs = flag.Bool("verbose_logs", false, "show all grpclog output, without filtering")

func noop() {}

// declareLogNoise declares that t is expected to emit the following noisy phrases,
// even on success. Those phrases will be filtered from grpclog output
// and only be shown if *verbose_logs or t ends up failing.
// The returned restore function should be called with defer to be run
// before the test ends.
func declareLogNoise(t *testing.T, phrases ...string) (restore func()) {
	if *verboseLogs {
		return noop
	}
	fw := &filterWriter{dst: os.Stderr, filter: phrases}
	testLogOutput.setWriter(fw)
	return func() {
		if t.Failed() {
			fw.mu.Lock()
			defer fw.mu.Unlock()
			if fw.buf.Len() > 0 {
				t.Logf("Complete log output:\n%s", fw.buf.Bytes())
			}
		}
		testLogOutput.setWriter(os.Stderr)
	}
}

type filterWriter struct {
	dst    io.Writer
	filter []string

	mu     sync.Mutex
	buf    bytes.Buffer
	wakeup chan<- bool // if non-nil, gets true on write
}

func (fw *filterWriter) Write(p []byte) (n int, err error) {
	fw.mu.Lock()
	fw.buf.Write(p)
	if fw.wakeup != nil {
		select {
		case fw.wakeup <- true:
		default:
		}
	}
	fw.mu.Unlock()

	ps := string(p)
	for _, f := range fw.filter {
		if strings.Contains(ps, f) {
			return len(p), nil
		}
	}
	return fw.dst.Write(p)
}

// stubServer is a server that is easy to customize within individual test
// cases.
type stubServer struct {
	// Guarantees we satisfy this interface; panics if unimplemented methods are called.
	testpb.TestServiceServer

	// Customizable implementations of server handlers.
	emptyCall      func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error)
	fullDuplexCall func(stream testpb.TestService_FullDuplexCallServer) error

	// A client connected to this service the test may use.  Created in Start().
	client testpb.TestServiceClient

	cleanups []func() // Lambdas executed in Stop(); populated by Start().
}

func (ss *stubServer) EmptyCall(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
	return ss.emptyCall(ctx, in)
}

func (ss *stubServer) FullDuplexCall(stream testpb.TestService_FullDuplexCallServer) error {
	return ss.fullDuplexCall(stream)
}

// Start starts the server and creates a client connected to it.
func (ss *stubServer) Start() error {
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		return fmt.Errorf(`net.Listen("tcp", ":0") = %v`, err)
	}
	ss.cleanups = append(ss.cleanups, func() { lis.Close() })

	s := grpc.NewServer()
	testpb.RegisterTestServiceServer(s, ss)
	go s.Serve(lis)
	ss.cleanups = append(ss.cleanups, s.Stop)

	cc, err := grpc.Dial(lis.Addr().String(), grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		return fmt.Errorf("grpc.Dial(%q) = %v", lis.Addr().String(), err)
	}
	ss.cleanups = append(ss.cleanups, func() { cc.Close() })

	ss.client = testpb.NewTestServiceClient(cc)
	return nil
}

func (ss *stubServer) Stop() {
	for i := len(ss.cleanups) - 1; i >= 0; i-- {
		ss.cleanups[i]()
	}
}

func TestUnaryProxyDoesNotForwardMetadata(t *testing.T) {
	const mdkey = "somedata"

	// endpoint ensures mdkey is NOT in metadata and returns an error if it is.
	endpoint := &stubServer{
		emptyCall: func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
			if md, ok := metadata.FromIncomingContext(ctx); !ok || md[mdkey] != nil {
				return nil, status.Errorf(codes.Internal, "endpoint: md=%v; want !contains(%q)", md, mdkey)
			}
			return &testpb.Empty{}, nil
		},
	}
	if err := endpoint.Start(); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer endpoint.Stop()

	// proxy ensures mdkey IS in metadata, then forwards the RPC to endpoint
	// without explicitly copying the metadata.
	proxy := &stubServer{
		emptyCall: func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
			if md, ok := metadata.FromIncomingContext(ctx); !ok || md[mdkey] == nil {
				return nil, status.Errorf(codes.Internal, "proxy: md=%v; want contains(%q)", md, mdkey)
			}
			return endpoint.client.EmptyCall(ctx, in)
		},
	}
	if err := proxy.Start(); err != nil {
		t.Fatalf("Error starting proxy server: %v", err)
	}
	defer proxy.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	md := metadata.Pairs(mdkey, "val")
	ctx = metadata.NewOutgoingContext(ctx, md)

	// Sanity check that endpoint properly errors when it sees mdkey.
	_, err := endpoint.client.EmptyCall(ctx, &testpb.Empty{})
	if s, ok := status.FromError(err); !ok || s.Code() != codes.Internal {
		t.Fatalf("endpoint.client.EmptyCall(_, _) = _, %v; want _, <status with Code()=Internal>", err)
	}

	if _, err := proxy.client.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Fatal(err.Error())
	}
}

func TestStreamingProxyDoesNotForwardMetadata(t *testing.T) {
	const mdkey = "somedata"

	// doFDC performs a FullDuplexCall with client and returns the error from the
	// first stream.Recv call, or nil if that error is io.EOF.  Calls t.Fatal if
	// the stream cannot be established.
	doFDC := func(ctx context.Context, client testpb.TestServiceClient) error {
		stream, err := client.FullDuplexCall(ctx)
		if err != nil {
			t.Fatalf("Unwanted error: %v", err)
		}
		if _, err := stream.Recv(); err != io.EOF {
			return err
		}
		return nil
	}

	// endpoint ensures mdkey is NOT in metadata and returns an error if it is.
	endpoint := &stubServer{
		fullDuplexCall: func(stream testpb.TestService_FullDuplexCallServer) error {
			ctx := stream.Context()
			if md, ok := metadata.FromIncomingContext(ctx); !ok || md[mdkey] != nil {
				return status.Errorf(codes.Internal, "endpoint: md=%v; want !contains(%q)", md, mdkey)
			}
			return nil
		},
	}
	if err := endpoint.Start(); err != nil {
		t.Fatalf("Error starting endpoint server: %v", err)
	}
	defer endpoint.Stop()

	// proxy ensures mdkey IS in metadata, then forwards the RPC to endpoint
	// without explicitly copying the metadata.
	proxy := &stubServer{
		fullDuplexCall: func(stream testpb.TestService_FullDuplexCallServer) error {
			ctx := stream.Context()
			if md, ok := metadata.FromIncomingContext(ctx); !ok || md[mdkey] == nil {
				return status.Errorf(codes.Internal, "endpoint: md=%v; want !contains(%q)", md, mdkey)
			}
			return doFDC(ctx, endpoint.client)
		},
	}
	if err := proxy.Start(); err != nil {
		t.Fatalf("Error starting proxy server: %v", err)
	}
	defer proxy.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	md := metadata.Pairs(mdkey, "val")
	ctx = metadata.NewOutgoingContext(ctx, md)

	// Sanity check that endpoint properly errors when it sees mdkey in ctx.
	err := doFDC(ctx, endpoint.client)
	if s, ok := status.FromError(err); !ok || s.Code() != codes.Internal {
		t.Fatalf("stream.Recv() = _, %v; want _, <status with Code()=Internal>", err)
	}

	if err := doFDC(ctx, proxy.client); err != nil {
		t.Fatalf("doFDC(_, proxy.client) = %v; want nil", err)
	}
}
