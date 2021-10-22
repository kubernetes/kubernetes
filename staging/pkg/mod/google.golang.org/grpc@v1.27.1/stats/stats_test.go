/*
 *
 * Copyright 2016 gRPC authors.
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

package stats_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	testpb "google.golang.org/grpc/stats/grpc_testing"
	"google.golang.org/grpc/status"
)

func init() {
	grpc.EnableTracing = false
}

type connCtxKey struct{}
type rpcCtxKey struct{}

var (
	// For headers sent to server:
	testMetadata = metadata.MD{
		"key1": []string{"value1"},
		"key2": []string{"value2"},
	}
	// For headers sent from server:
	testHeaderMetadata = metadata.MD{
		"hkey1": []string{"headerValue1"},
		"hkey2": []string{"headerValue2"},
	}
	// For trailers sent from server:
	testTrailerMetadata = metadata.MD{
		"tkey1": []string{"trailerValue1"},
		"tkey2": []string{"trailerValue2"},
	}
	// The id for which the service handler should return error.
	errorID int32 = 32202
)

type testServer struct {
	testpb.UnimplementedTestServiceServer
}

func (s *testServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	if err := grpc.SendHeader(ctx, testHeaderMetadata); err != nil {
		return nil, status.Errorf(status.Code(err), "grpc.SendHeader(_, %v) = %v, want <nil>", testHeaderMetadata, err)
	}
	if err := grpc.SetTrailer(ctx, testTrailerMetadata); err != nil {
		return nil, status.Errorf(status.Code(err), "grpc.SetTrailer(_, %v) = %v, want <nil>", testTrailerMetadata, err)
	}

	if in.Id == errorID {
		return nil, fmt.Errorf("got error id: %v", in.Id)
	}

	return &testpb.SimpleResponse{Id: in.Id}, nil
}

func (s *testServer) FullDuplexCall(stream testpb.TestService_FullDuplexCallServer) error {
	if err := stream.SendHeader(testHeaderMetadata); err != nil {
		return status.Errorf(status.Code(err), "%v.SendHeader(%v) = %v, want %v", stream, testHeaderMetadata, err, nil)
	}
	stream.SetTrailer(testTrailerMetadata)
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			return nil
		}
		if err != nil {
			return err
		}

		if in.Id == errorID {
			return fmt.Errorf("got error id: %v", in.Id)
		}

		if err := stream.Send(&testpb.SimpleResponse{Id: in.Id}); err != nil {
			return err
		}
	}
}

func (s *testServer) ClientStreamCall(stream testpb.TestService_ClientStreamCallServer) error {
	if err := stream.SendHeader(testHeaderMetadata); err != nil {
		return status.Errorf(status.Code(err), "%v.SendHeader(%v) = %v, want %v", stream, testHeaderMetadata, err, nil)
	}
	stream.SetTrailer(testTrailerMetadata)
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			// read done.
			return stream.SendAndClose(&testpb.SimpleResponse{Id: int32(0)})
		}
		if err != nil {
			return err
		}

		if in.Id == errorID {
			return fmt.Errorf("got error id: %v", in.Id)
		}
	}
}

func (s *testServer) ServerStreamCall(in *testpb.SimpleRequest, stream testpb.TestService_ServerStreamCallServer) error {
	if err := stream.SendHeader(testHeaderMetadata); err != nil {
		return status.Errorf(status.Code(err), "%v.SendHeader(%v) = %v, want %v", stream, testHeaderMetadata, err, nil)
	}
	stream.SetTrailer(testTrailerMetadata)

	if in.Id == errorID {
		return fmt.Errorf("got error id: %v", in.Id)
	}

	for i := 0; i < 5; i++ {
		if err := stream.Send(&testpb.SimpleResponse{Id: in.Id}); err != nil {
			return err
		}
	}
	return nil
}

// test is an end-to-end test. It should be created with the newTest
// func, modified as needed, and then started with its startServer method.
// It should be cleaned up with the tearDown method.
type test struct {
	t                  *testing.T
	compress           string
	clientStatsHandler stats.Handler
	serverStatsHandler stats.Handler

	testServer testpb.TestServiceServer // nil means none
	// srv and srvAddr are set once startServer is called.
	srv     *grpc.Server
	srvAddr string

	cc *grpc.ClientConn // nil until requested via clientConn
}

func (te *test) tearDown() {
	if te.cc != nil {
		te.cc.Close()
		te.cc = nil
	}
	te.srv.Stop()
}

type testConfig struct {
	compress string
}

// newTest returns a new test using the provided testing.T and
// environment.  It is returned with default values. Tests should
// modify it before calling its startServer and clientConn methods.
func newTest(t *testing.T, tc *testConfig, ch stats.Handler, sh stats.Handler) *test {
	te := &test{
		t:                  t,
		compress:           tc.compress,
		clientStatsHandler: ch,
		serverStatsHandler: sh,
	}
	return te
}

// startServer starts a gRPC server listening. Callers should defer a
// call to te.tearDown to clean up.
func (te *test) startServer(ts testpb.TestServiceServer) {
	te.testServer = ts
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		te.t.Fatalf("Failed to listen: %v", err)
	}
	var opts []grpc.ServerOption
	if te.compress == "gzip" {
		opts = append(opts,
			grpc.RPCCompressor(grpc.NewGZIPCompressor()),
			grpc.RPCDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	if te.serverStatsHandler != nil {
		opts = append(opts, grpc.StatsHandler(te.serverStatsHandler))
	}
	s := grpc.NewServer(opts...)
	te.srv = s
	if te.testServer != nil {
		testpb.RegisterTestServiceServer(s, te.testServer)
	}

	go s.Serve(lis)
	te.srvAddr = lis.Addr().String()
}

func (te *test) clientConn() *grpc.ClientConn {
	if te.cc != nil {
		return te.cc
	}
	opts := []grpc.DialOption{grpc.WithInsecure(), grpc.WithBlock()}
	if te.compress == "gzip" {
		opts = append(opts,
			grpc.WithCompressor(grpc.NewGZIPCompressor()),
			grpc.WithDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	if te.clientStatsHandler != nil {
		opts = append(opts, grpc.WithStatsHandler(te.clientStatsHandler))
	}

	var err error
	te.cc, err = grpc.Dial(te.srvAddr, opts...)
	if err != nil {
		te.t.Fatalf("Dial(%q) = %v", te.srvAddr, err)
	}
	return te.cc
}

type rpcType int

const (
	unaryRPC rpcType = iota
	clientStreamRPC
	serverStreamRPC
	fullDuplexStreamRPC
)

type rpcConfig struct {
	count    int  // Number of requests and responses for streaming RPCs.
	success  bool // Whether the RPC should succeed or return error.
	failfast bool
	callType rpcType // Type of RPC.
}

func (te *test) doUnaryCall(c *rpcConfig) (*testpb.SimpleRequest, *testpb.SimpleResponse, error) {
	var (
		resp *testpb.SimpleResponse
		req  *testpb.SimpleRequest
		err  error
	)
	tc := testpb.NewTestServiceClient(te.clientConn())
	if c.success {
		req = &testpb.SimpleRequest{Id: errorID + 1}
	} else {
		req = &testpb.SimpleRequest{Id: errorID}
	}
	ctx := metadata.NewOutgoingContext(context.Background(), testMetadata)
	resp, err = tc.UnaryCall(ctx, req, grpc.WaitForReady(!c.failfast))
	return req, resp, err
}

func (te *test) doFullDuplexCallRoundtrip(c *rpcConfig) ([]*testpb.SimpleRequest, []*testpb.SimpleResponse, error) {
	var (
		reqs  []*testpb.SimpleRequest
		resps []*testpb.SimpleResponse
		err   error
	)
	tc := testpb.NewTestServiceClient(te.clientConn())
	stream, err := tc.FullDuplexCall(metadata.NewOutgoingContext(context.Background(), testMetadata), grpc.WaitForReady(!c.failfast))
	if err != nil {
		return reqs, resps, err
	}
	var startID int32
	if !c.success {
		startID = errorID
	}
	for i := 0; i < c.count; i++ {
		req := &testpb.SimpleRequest{
			Id: int32(i) + startID,
		}
		reqs = append(reqs, req)
		if err = stream.Send(req); err != nil {
			return reqs, resps, err
		}
		var resp *testpb.SimpleResponse
		if resp, err = stream.Recv(); err != nil {
			return reqs, resps, err
		}
		resps = append(resps, resp)
	}
	if err = stream.CloseSend(); err != nil && err != io.EOF {
		return reqs, resps, err
	}
	if _, err = stream.Recv(); err != io.EOF {
		return reqs, resps, err
	}

	return reqs, resps, nil
}

func (te *test) doClientStreamCall(c *rpcConfig) ([]*testpb.SimpleRequest, *testpb.SimpleResponse, error) {
	var (
		reqs []*testpb.SimpleRequest
		resp *testpb.SimpleResponse
		err  error
	)
	tc := testpb.NewTestServiceClient(te.clientConn())
	stream, err := tc.ClientStreamCall(metadata.NewOutgoingContext(context.Background(), testMetadata), grpc.WaitForReady(!c.failfast))
	if err != nil {
		return reqs, resp, err
	}
	var startID int32
	if !c.success {
		startID = errorID
	}
	for i := 0; i < c.count; i++ {
		req := &testpb.SimpleRequest{
			Id: int32(i) + startID,
		}
		reqs = append(reqs, req)
		if err = stream.Send(req); err != nil {
			return reqs, resp, err
		}
	}
	resp, err = stream.CloseAndRecv()
	return reqs, resp, err
}

func (te *test) doServerStreamCall(c *rpcConfig) (*testpb.SimpleRequest, []*testpb.SimpleResponse, error) {
	var (
		req   *testpb.SimpleRequest
		resps []*testpb.SimpleResponse
		err   error
	)

	tc := testpb.NewTestServiceClient(te.clientConn())

	var startID int32
	if !c.success {
		startID = errorID
	}
	req = &testpb.SimpleRequest{Id: startID}
	stream, err := tc.ServerStreamCall(metadata.NewOutgoingContext(context.Background(), testMetadata), req, grpc.WaitForReady(!c.failfast))
	if err != nil {
		return req, resps, err
	}
	for {
		var resp *testpb.SimpleResponse
		resp, err := stream.Recv()
		if err == io.EOF {
			return req, resps, nil
		} else if err != nil {
			return req, resps, err
		}
		resps = append(resps, resp)
	}
}

type expectedData struct {
	method      string
	serverAddr  string
	compression string
	reqIdx      int
	requests    []*testpb.SimpleRequest
	respIdx     int
	responses   []*testpb.SimpleResponse
	err         error
	failfast    bool
}

type gotData struct {
	ctx    context.Context
	client bool
	s      interface{} // This could be RPCStats or ConnStats.
}

const (
	begin int = iota
	end
	inPayload
	inHeader
	inTrailer
	outPayload
	outHeader
	// TODO: test outTrailer ?
	connBegin
	connEnd
)

func checkBegin(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.Begin
	)
	if st, ok = d.s.(*stats.Begin); !ok {
		t.Fatalf("got %T, want Begin", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if st.BeginTime.IsZero() {
		t.Fatalf("st.BeginTime = %v, want <non-zero>", st.BeginTime)
	}
	if d.client {
		if st.FailFast != e.failfast {
			t.Fatalf("st.FailFast = %v, want %v", st.FailFast, e.failfast)
		}
	}
}

func checkInHeader(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.InHeader
	)
	if st, ok = d.s.(*stats.InHeader); !ok {
		t.Fatalf("got %T, want InHeader", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if d.client {
		// additional headers might be injected so instead of testing equality, test that all the
		// expected headers keys have the expected header values.
		for key := range testHeaderMetadata {
			if !reflect.DeepEqual(st.Header.Get(key), testHeaderMetadata.Get(key)) {
				t.Fatalf("st.Header[%s] = %v, want %v", key, st.Header.Get(key), testHeaderMetadata.Get(key))
			}
		}
	} else {
		if st.FullMethod != e.method {
			t.Fatalf("st.FullMethod = %s, want %v", st.FullMethod, e.method)
		}
		if st.LocalAddr.String() != e.serverAddr {
			t.Fatalf("st.LocalAddr = %v, want %v", st.LocalAddr, e.serverAddr)
		}
		if st.Compression != e.compression {
			t.Fatalf("st.Compression = %v, want %v", st.Compression, e.compression)
		}
		// additional headers might be injected so instead of testing equality, test that all the
		// expected headers keys have the expected header values.
		for key := range testMetadata {
			if !reflect.DeepEqual(st.Header.Get(key), testMetadata.Get(key)) {
				t.Fatalf("st.Header[%s] = %v, want %v", key, st.Header.Get(key), testMetadata.Get(key))
			}
		}

		if connInfo, ok := d.ctx.Value(connCtxKey{}).(*stats.ConnTagInfo); ok {
			if connInfo.RemoteAddr != st.RemoteAddr {
				t.Fatalf("connInfo.RemoteAddr = %v, want %v", connInfo.RemoteAddr, st.RemoteAddr)
			}
			if connInfo.LocalAddr != st.LocalAddr {
				t.Fatalf("connInfo.LocalAddr = %v, want %v", connInfo.LocalAddr, st.LocalAddr)
			}
		} else {
			t.Fatalf("got context %v, want one with connCtxKey", d.ctx)
		}
		if rpcInfo, ok := d.ctx.Value(rpcCtxKey{}).(*stats.RPCTagInfo); ok {
			if rpcInfo.FullMethodName != st.FullMethod {
				t.Fatalf("rpcInfo.FullMethod = %s, want %v", rpcInfo.FullMethodName, st.FullMethod)
			}
		} else {
			t.Fatalf("got context %v, want one with rpcCtxKey", d.ctx)
		}
	}
}

func checkInPayload(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.InPayload
	)
	if st, ok = d.s.(*stats.InPayload); !ok {
		t.Fatalf("got %T, want InPayload", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if d.client {
		b, err := proto.Marshal(e.responses[e.respIdx])
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if reflect.TypeOf(st.Payload) != reflect.TypeOf(e.responses[e.respIdx]) {
			t.Fatalf("st.Payload = %T, want %T", st.Payload, e.responses[e.respIdx])
		}
		e.respIdx++
		if string(st.Data) != string(b) {
			t.Fatalf("st.Data = %v, want %v", st.Data, b)
		}
		if st.Length != len(b) {
			t.Fatalf("st.Lenght = %v, want %v", st.Length, len(b))
		}
	} else {
		b, err := proto.Marshal(e.requests[e.reqIdx])
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if reflect.TypeOf(st.Payload) != reflect.TypeOf(e.requests[e.reqIdx]) {
			t.Fatalf("st.Payload = %T, want %T", st.Payload, e.requests[e.reqIdx])
		}
		e.reqIdx++
		if string(st.Data) != string(b) {
			t.Fatalf("st.Data = %v, want %v", st.Data, b)
		}
		if st.Length != len(b) {
			t.Fatalf("st.Lenght = %v, want %v", st.Length, len(b))
		}
	}
	// Below are sanity checks that WireLength and RecvTime are populated.
	// TODO: check values of WireLength and RecvTime.
	if len(st.Data) > 0 && st.WireLength == 0 {
		t.Fatalf("st.WireLength = %v with non-empty data, want <non-zero>",
			st.WireLength)
	}
	if st.RecvTime.IsZero() {
		t.Fatalf("st.ReceivedTime = %v, want <non-zero>", st.RecvTime)
	}
}

func checkInTrailer(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.InTrailer
	)
	if st, ok = d.s.(*stats.InTrailer); !ok {
		t.Fatalf("got %T, want InTrailer", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if !st.Client {
		t.Fatalf("st IsClient = false, want true")
	}
	if !reflect.DeepEqual(st.Trailer, testTrailerMetadata) {
		t.Fatalf("st.Trailer = %v, want %v", st.Trailer, testTrailerMetadata)
	}
}

func checkOutHeader(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.OutHeader
	)
	if st, ok = d.s.(*stats.OutHeader); !ok {
		t.Fatalf("got %T, want OutHeader", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if d.client {
		if st.FullMethod != e.method {
			t.Fatalf("st.FullMethod = %s, want %v", st.FullMethod, e.method)
		}
		if st.RemoteAddr.String() != e.serverAddr {
			t.Fatalf("st.RemoteAddr = %v, want %v", st.RemoteAddr, e.serverAddr)
		}
		if st.Compression != e.compression {
			t.Fatalf("st.Compression = %v, want %v", st.Compression, e.compression)
		}
		// additional headers might be injected so instead of testing equality, test that all the
		// expected headers keys have the expected header values.
		for key := range testMetadata {
			if !reflect.DeepEqual(st.Header.Get(key), testMetadata.Get(key)) {
				t.Fatalf("st.Header[%s] = %v, want %v", key, st.Header.Get(key), testMetadata.Get(key))
			}
		}

		if rpcInfo, ok := d.ctx.Value(rpcCtxKey{}).(*stats.RPCTagInfo); ok {
			if rpcInfo.FullMethodName != st.FullMethod {
				t.Fatalf("rpcInfo.FullMethod = %s, want %v", rpcInfo.FullMethodName, st.FullMethod)
			}
		} else {
			t.Fatalf("got context %v, want one with rpcCtxKey", d.ctx)
		}
	} else {
		// additional headers might be injected so instead of testing equality, test that all the
		// expected headers keys have the expected header values.
		for key := range testHeaderMetadata {
			if !reflect.DeepEqual(st.Header.Get(key), testHeaderMetadata.Get(key)) {
				t.Fatalf("st.Header[%s] = %v, want %v", key, st.Header.Get(key), testHeaderMetadata.Get(key))
			}
		}
	}
}

func checkOutPayload(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.OutPayload
	)
	if st, ok = d.s.(*stats.OutPayload); !ok {
		t.Fatalf("got %T, want OutPayload", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if d.client {
		b, err := proto.Marshal(e.requests[e.reqIdx])
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if reflect.TypeOf(st.Payload) != reflect.TypeOf(e.requests[e.reqIdx]) {
			t.Fatalf("st.Payload = %T, want %T", st.Payload, e.requests[e.reqIdx])
		}
		e.reqIdx++
		if string(st.Data) != string(b) {
			t.Fatalf("st.Data = %v, want %v", st.Data, b)
		}
		if st.Length != len(b) {
			t.Fatalf("st.Lenght = %v, want %v", st.Length, len(b))
		}
	} else {
		b, err := proto.Marshal(e.responses[e.respIdx])
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if reflect.TypeOf(st.Payload) != reflect.TypeOf(e.responses[e.respIdx]) {
			t.Fatalf("st.Payload = %T, want %T", st.Payload, e.responses[e.respIdx])
		}
		e.respIdx++
		if string(st.Data) != string(b) {
			t.Fatalf("st.Data = %v, want %v", st.Data, b)
		}
		if st.Length != len(b) {
			t.Fatalf("st.Lenght = %v, want %v", st.Length, len(b))
		}
	}
	// Below are sanity checks that WireLength and SentTime are populated.
	// TODO: check values of WireLength and SentTime.
	if len(st.Data) > 0 && st.WireLength == 0 {
		t.Fatalf("st.WireLength = %v with non-empty data, want <non-zero>",
			st.WireLength)
	}
	if st.SentTime.IsZero() {
		t.Fatalf("st.SentTime = %v, want <non-zero>", st.SentTime)
	}
}

func checkOutTrailer(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.OutTrailer
	)
	if st, ok = d.s.(*stats.OutTrailer); !ok {
		t.Fatalf("got %T, want OutTrailer", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if st.Client {
		t.Fatalf("st IsClient = true, want false")
	}
	if !reflect.DeepEqual(st.Trailer, testTrailerMetadata) {
		t.Fatalf("st.Trailer = %v, want %v", st.Trailer, testTrailerMetadata)
	}
}

func checkEnd(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.End
	)
	if st, ok = d.s.(*stats.End); !ok {
		t.Fatalf("got %T, want End", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	if st.BeginTime.IsZero() {
		t.Fatalf("st.BeginTime = %v, want <non-zero>", st.BeginTime)
	}
	if st.EndTime.IsZero() {
		t.Fatalf("st.EndTime = %v, want <non-zero>", st.EndTime)
	}

	actual, ok := status.FromError(st.Error)
	if !ok {
		t.Fatalf("expected st.Error to be a statusError, got %v (type %T)", st.Error, st.Error)
	}

	expectedStatus, _ := status.FromError(e.err)
	if actual.Code() != expectedStatus.Code() || actual.Message() != expectedStatus.Message() {
		t.Fatalf("st.Error = %v, want %v", st.Error, e.err)
	}

	if st.Client {
		if !reflect.DeepEqual(st.Trailer, testTrailerMetadata) {
			t.Fatalf("st.Trailer = %v, want %v", st.Trailer, testTrailerMetadata)
		}
	} else {
		if st.Trailer != nil {
			t.Fatalf("st.Trailer = %v, want nil", st.Trailer)
		}
	}
}

func checkConnBegin(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.ConnBegin
	)
	if st, ok = d.s.(*stats.ConnBegin); !ok {
		t.Fatalf("got %T, want ConnBegin", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	st.IsClient() // TODO remove this.
}

func checkConnEnd(t *testing.T, d *gotData, e *expectedData) {
	var (
		ok bool
		st *stats.ConnEnd
	)
	if st, ok = d.s.(*stats.ConnEnd); !ok {
		t.Fatalf("got %T, want ConnEnd", d.s)
	}
	if d.ctx == nil {
		t.Fatalf("d.ctx = nil, want <non-nil>")
	}
	st.IsClient() // TODO remove this.
}

type statshandler struct {
	mu      sync.Mutex
	gotRPC  []*gotData
	gotConn []*gotData
}

func (h *statshandler) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context {
	return context.WithValue(ctx, connCtxKey{}, info)
}

func (h *statshandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	return context.WithValue(ctx, rpcCtxKey{}, info)
}

func (h *statshandler) HandleConn(ctx context.Context, s stats.ConnStats) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.gotConn = append(h.gotConn, &gotData{ctx, s.IsClient(), s})
}

func (h *statshandler) HandleRPC(ctx context.Context, s stats.RPCStats) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.gotRPC = append(h.gotRPC, &gotData{ctx, s.IsClient(), s})
}

func checkConnStats(t *testing.T, got []*gotData) {
	if len(got) <= 0 || len(got)%2 != 0 {
		for i, g := range got {
			t.Errorf(" - %v, %T = %+v, ctx: %v", i, g.s, g.s, g.ctx)
		}
		t.Fatalf("got %v stats, want even positive number", len(got))
	}
	// The first conn stats must be a ConnBegin.
	checkConnBegin(t, got[0], nil)
	// The last conn stats must be a ConnEnd.
	checkConnEnd(t, got[len(got)-1], nil)
}

func checkServerStats(t *testing.T, got []*gotData, expect *expectedData, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	if len(got) != len(checkFuncs) {
		for i, g := range got {
			t.Errorf(" - %v, %T", i, g.s)
		}
		t.Fatalf("got %v stats, want %v stats", len(got), len(checkFuncs))
	}

	var rpcctx context.Context
	for i := 0; i < len(got); i++ {
		if _, ok := got[i].s.(stats.RPCStats); ok {
			if rpcctx != nil && got[i].ctx != rpcctx {
				t.Fatalf("got different contexts with stats %T", got[i].s)
			}
			rpcctx = got[i].ctx
		}
	}

	for i, f := range checkFuncs {
		f(t, got[i], expect)
	}
}

func testServerStats(t *testing.T, tc *testConfig, cc *rpcConfig, checkFuncs []func(t *testing.T, d *gotData, e *expectedData)) {
	h := &statshandler{}
	te := newTest(t, tc, nil, h)
	te.startServer(&testServer{})
	defer te.tearDown()

	var (
		reqs   []*testpb.SimpleRequest
		resps  []*testpb.SimpleResponse
		err    error
		method string

		req  *testpb.SimpleRequest
		resp *testpb.SimpleResponse
		e    error
	)

	switch cc.callType {
	case unaryRPC:
		method = "/grpc.testing.TestService/UnaryCall"
		req, resp, e = te.doUnaryCall(cc)
		reqs = []*testpb.SimpleRequest{req}
		resps = []*testpb.SimpleResponse{resp}
		err = e
	case clientStreamRPC:
		method = "/grpc.testing.TestService/ClientStreamCall"
		reqs, resp, e = te.doClientStreamCall(cc)
		resps = []*testpb.SimpleResponse{resp}
		err = e
	case serverStreamRPC:
		method = "/grpc.testing.TestService/ServerStreamCall"
		req, resps, e = te.doServerStreamCall(cc)
		reqs = []*testpb.SimpleRequest{req}
		err = e
	case fullDuplexStreamRPC:
		method = "/grpc.testing.TestService/FullDuplexCall"
		reqs, resps, err = te.doFullDuplexCallRoundtrip(cc)
	}
	if cc.success != (err == nil) {
		t.Fatalf("cc.success: %v, got error: %v", cc.success, err)
	}
	te.cc.Close()
	te.srv.GracefulStop() // Wait for the server to stop.

	for {
		h.mu.Lock()
		if len(h.gotRPC) >= len(checkFuncs) {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	for {
		h.mu.Lock()
		if _, ok := h.gotConn[len(h.gotConn)-1].s.(*stats.ConnEnd); ok {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	expect := &expectedData{
		serverAddr:  te.srvAddr,
		compression: tc.compress,
		method:      method,
		requests:    reqs,
		responses:   resps,
		err:         err,
	}

	h.mu.Lock()
	checkConnStats(t, h.gotConn)
	h.mu.Unlock()
	checkServerStats(t, h.gotRPC, expect, checkFuncs)
}

func TestServerStatsUnaryRPC(t *testing.T) {
	testServerStats(t, &testConfig{compress: ""}, &rpcConfig{success: true, callType: unaryRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkInPayload,
		checkOutHeader,
		checkOutPayload,
		checkOutTrailer,
		checkEnd,
	})
}

func TestServerStatsUnaryRPCError(t *testing.T) {
	testServerStats(t, &testConfig{compress: ""}, &rpcConfig{success: false, callType: unaryRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkInPayload,
		checkOutHeader,
		checkOutTrailer,
		checkEnd,
	})
}

func TestServerStatsClientStreamRPC(t *testing.T) {
	count := 5
	checkFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
	}
	ioPayFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInPayload,
	}
	for i := 0; i < count; i++ {
		checkFuncs = append(checkFuncs, ioPayFuncs...)
	}
	checkFuncs = append(checkFuncs,
		checkOutPayload,
		checkOutTrailer,
		checkEnd,
	)
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, callType: clientStreamRPC}, checkFuncs)
}

func TestServerStatsClientStreamRPCError(t *testing.T) {
	count := 1
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, callType: clientStreamRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
		checkInPayload,
		checkOutTrailer,
		checkEnd,
	})
}

func TestServerStatsServerStreamRPC(t *testing.T) {
	count := 5
	checkFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkInPayload,
		checkOutHeader,
	}
	ioPayFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkOutPayload,
	}
	for i := 0; i < count; i++ {
		checkFuncs = append(checkFuncs, ioPayFuncs...)
	}
	checkFuncs = append(checkFuncs,
		checkOutTrailer,
		checkEnd,
	)
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, callType: serverStreamRPC}, checkFuncs)
}

func TestServerStatsServerStreamRPCError(t *testing.T) {
	count := 5
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, callType: serverStreamRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkInPayload,
		checkOutHeader,
		checkOutTrailer,
		checkEnd,
	})
}

func TestServerStatsFullDuplexRPC(t *testing.T) {
	count := 5
	checkFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
	}
	ioPayFuncs := []func(t *testing.T, d *gotData, e *expectedData){
		checkInPayload,
		checkOutPayload,
	}
	for i := 0; i < count; i++ {
		checkFuncs = append(checkFuncs, ioPayFuncs...)
	}
	checkFuncs = append(checkFuncs,
		checkOutTrailer,
		checkEnd,
	)
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, callType: fullDuplexStreamRPC}, checkFuncs)
}

func TestServerStatsFullDuplexRPCError(t *testing.T) {
	count := 5
	testServerStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, callType: fullDuplexStreamRPC}, []func(t *testing.T, d *gotData, e *expectedData){
		checkInHeader,
		checkBegin,
		checkOutHeader,
		checkInPayload,
		checkOutTrailer,
		checkEnd,
	})
}

type checkFuncWithCount struct {
	f func(t *testing.T, d *gotData, e *expectedData)
	c int // expected count
}

func checkClientStats(t *testing.T, got []*gotData, expect *expectedData, checkFuncs map[int]*checkFuncWithCount) {
	var expectLen int
	for _, v := range checkFuncs {
		expectLen += v.c
	}
	if len(got) != expectLen {
		for i, g := range got {
			t.Errorf(" - %v, %T", i, g.s)
		}
		t.Fatalf("got %v stats, want %v stats", len(got), expectLen)
	}

	var tagInfoInCtx *stats.RPCTagInfo
	for i := 0; i < len(got); i++ {
		if _, ok := got[i].s.(stats.RPCStats); ok {
			tagInfoInCtxNew, _ := got[i].ctx.Value(rpcCtxKey{}).(*stats.RPCTagInfo)
			if tagInfoInCtx != nil && tagInfoInCtx != tagInfoInCtxNew {
				t.Fatalf("got context containing different tagInfo with stats %T", got[i].s)
			}
			tagInfoInCtx = tagInfoInCtxNew
		}
	}

	for _, s := range got {
		switch s.s.(type) {
		case *stats.Begin:
			if checkFuncs[begin].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[begin].f(t, s, expect)
			checkFuncs[begin].c--
		case *stats.OutHeader:
			if checkFuncs[outHeader].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[outHeader].f(t, s, expect)
			checkFuncs[outHeader].c--
		case *stats.OutPayload:
			if checkFuncs[outPayload].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[outPayload].f(t, s, expect)
			checkFuncs[outPayload].c--
		case *stats.InHeader:
			if checkFuncs[inHeader].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inHeader].f(t, s, expect)
			checkFuncs[inHeader].c--
		case *stats.InPayload:
			if checkFuncs[inPayload].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inPayload].f(t, s, expect)
			checkFuncs[inPayload].c--
		case *stats.InTrailer:
			if checkFuncs[inTrailer].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[inTrailer].f(t, s, expect)
			checkFuncs[inTrailer].c--
		case *stats.End:
			if checkFuncs[end].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[end].f(t, s, expect)
			checkFuncs[end].c--
		case *stats.ConnBegin:
			if checkFuncs[connBegin].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[connBegin].f(t, s, expect)
			checkFuncs[connBegin].c--
		case *stats.ConnEnd:
			if checkFuncs[connEnd].c <= 0 {
				t.Fatalf("unexpected stats: %T", s.s)
			}
			checkFuncs[connEnd].f(t, s, expect)
			checkFuncs[connEnd].c--
		default:
			t.Fatalf("unexpected stats: %T", s.s)
		}
	}
}

func testClientStats(t *testing.T, tc *testConfig, cc *rpcConfig, checkFuncs map[int]*checkFuncWithCount) {
	h := &statshandler{}
	te := newTest(t, tc, h, nil)
	te.startServer(&testServer{})
	defer te.tearDown()

	var (
		reqs   []*testpb.SimpleRequest
		resps  []*testpb.SimpleResponse
		method string
		err    error

		req  *testpb.SimpleRequest
		resp *testpb.SimpleResponse
		e    error
	)
	switch cc.callType {
	case unaryRPC:
		method = "/grpc.testing.TestService/UnaryCall"
		req, resp, e = te.doUnaryCall(cc)
		reqs = []*testpb.SimpleRequest{req}
		resps = []*testpb.SimpleResponse{resp}
		err = e
	case clientStreamRPC:
		method = "/grpc.testing.TestService/ClientStreamCall"
		reqs, resp, e = te.doClientStreamCall(cc)
		resps = []*testpb.SimpleResponse{resp}
		err = e
	case serverStreamRPC:
		method = "/grpc.testing.TestService/ServerStreamCall"
		req, resps, e = te.doServerStreamCall(cc)
		reqs = []*testpb.SimpleRequest{req}
		err = e
	case fullDuplexStreamRPC:
		method = "/grpc.testing.TestService/FullDuplexCall"
		reqs, resps, err = te.doFullDuplexCallRoundtrip(cc)
	}
	if cc.success != (err == nil) {
		t.Fatalf("cc.success: %v, got error: %v", cc.success, err)
	}
	te.cc.Close()
	te.srv.GracefulStop() // Wait for the server to stop.

	lenRPCStats := 0
	for _, v := range checkFuncs {
		lenRPCStats += v.c
	}
	for {
		h.mu.Lock()
		if len(h.gotRPC) >= lenRPCStats {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	for {
		h.mu.Lock()
		if _, ok := h.gotConn[len(h.gotConn)-1].s.(*stats.ConnEnd); ok {
			h.mu.Unlock()
			break
		}
		h.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	expect := &expectedData{
		serverAddr:  te.srvAddr,
		compression: tc.compress,
		method:      method,
		requests:    reqs,
		responses:   resps,
		failfast:    cc.failfast,
		err:         err,
	}

	h.mu.Lock()
	checkConnStats(t, h.gotConn)
	h.mu.Unlock()
	checkClientStats(t, h.gotRPC, expect, checkFuncs)
}

func TestClientStatsUnaryRPC(t *testing.T) {
	testClientStats(t, &testConfig{compress: ""}, &rpcConfig{success: true, failfast: false, callType: unaryRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inPayload:  {checkInPayload, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsUnaryRPCError(t *testing.T) {
	testClientStats(t, &testConfig{compress: ""}, &rpcConfig{success: false, failfast: false, callType: unaryRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsClientStreamRPC(t *testing.T) {
	count := 5
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, failfast: false, callType: clientStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		inHeader:   {checkInHeader, 1},
		outPayload: {checkOutPayload, count},
		inTrailer:  {checkInTrailer, 1},
		inPayload:  {checkInPayload, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsClientStreamRPCError(t *testing.T) {
	count := 1
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, failfast: false, callType: clientStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		inHeader:   {checkInHeader, 1},
		outPayload: {checkOutPayload, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsServerStreamRPC(t *testing.T) {
	count := 5
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, failfast: false, callType: serverStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inPayload:  {checkInPayload, count},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsServerStreamRPCError(t *testing.T) {
	count := 5
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, failfast: false, callType: serverStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsFullDuplexRPC(t *testing.T) {
	count := 5
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: true, failfast: false, callType: fullDuplexStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, count},
		inHeader:   {checkInHeader, 1},
		inPayload:  {checkInPayload, count},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestClientStatsFullDuplexRPCError(t *testing.T) {
	count := 5
	testClientStats(t, &testConfig{compress: "gzip"}, &rpcConfig{count: count, success: false, failfast: false, callType: fullDuplexStreamRPC}, map[int]*checkFuncWithCount{
		begin:      {checkBegin, 1},
		outHeader:  {checkOutHeader, 1},
		outPayload: {checkOutPayload, 1},
		inHeader:   {checkInHeader, 1},
		inTrailer:  {checkInTrailer, 1},
		end:        {checkEnd, 1},
	})
}

func TestTags(t *testing.T) {
	b := []byte{5, 2, 4, 3, 1}
	ctx := stats.SetTags(context.Background(), b)
	if tg := stats.OutgoingTags(ctx); !reflect.DeepEqual(tg, b) {
		t.Errorf("OutgoingTags(%v) = %v; want %v", ctx, tg, b)
	}
	if tg := stats.Tags(ctx); tg != nil {
		t.Errorf("Tags(%v) = %v; want nil", ctx, tg)
	}

	ctx = stats.SetIncomingTags(context.Background(), b)
	if tg := stats.Tags(ctx); !reflect.DeepEqual(tg, b) {
		t.Errorf("Tags(%v) = %v; want %v", ctx, tg, b)
	}
	if tg := stats.OutgoingTags(ctx); tg != nil {
		t.Errorf("OutgoingTags(%v) = %v; want nil", ctx, tg)
	}
}

func TestTrace(t *testing.T) {
	b := []byte{5, 2, 4, 3, 1}
	ctx := stats.SetTrace(context.Background(), b)
	if tr := stats.OutgoingTrace(ctx); !reflect.DeepEqual(tr, b) {
		t.Errorf("OutgoingTrace(%v) = %v; want %v", ctx, tr, b)
	}
	if tr := stats.Trace(ctx); tr != nil {
		t.Errorf("Trace(%v) = %v; want nil", ctx, tr)
	}

	ctx = stats.SetIncomingTrace(context.Background(), b)
	if tr := stats.Trace(ctx); !reflect.DeepEqual(tr, b) {
		t.Errorf("Trace(%v) = %v; want %v", ctx, tr, b)
	}
	if tr := stats.OutgoingTrace(ctx); tr != nil {
		t.Errorf("OutgoingTrace(%v) = %v; want nil", ctx, tr)
	}
}
