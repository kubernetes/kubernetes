// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

package grpc_prometheus

import (
	"net"
	"testing"

	"time"

	"io"

	pb_testproto "github.com/grpc-ecosystem/go-grpc-prometheus/examples/testproto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func TestClientInterceptorSuite(t *testing.T) {
	suite.Run(t, &ClientInterceptorTestSuite{})
}

type ClientInterceptorTestSuite struct {
	suite.Suite

	serverListener net.Listener
	server         *grpc.Server
	clientConn     *grpc.ClientConn
	testClient     pb_testproto.TestServiceClient
	ctx            context.Context
}

func (s *ClientInterceptorTestSuite) SetupSuite() {
	var err error

	EnableClientHandlingTimeHistogram()

	s.serverListener, err = net.Listen("tcp", "127.0.0.1:0")
	require.NoError(s.T(), err, "must be able to allocate a port for serverListener")

	// This is the point where we hook up the interceptor
	s.server = grpc.NewServer()
	pb_testproto.RegisterTestServiceServer(s.server, &testService{t: s.T()})

	go func() {
		s.server.Serve(s.serverListener)
	}()

	s.clientConn, err = grpc.Dial(
		s.serverListener.Addr().String(),
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithUnaryInterceptor(UnaryClientInterceptor),
		grpc.WithStreamInterceptor(StreamClientInterceptor),
		grpc.WithTimeout(2*time.Second))
	require.NoError(s.T(), err, "must not error on client Dial")
	s.testClient = pb_testproto.NewTestServiceClient(s.clientConn)
}

func (s *ClientInterceptorTestSuite) SetupTest() {
	// Make all RPC calls last at most 2 sec, meaning all async issues or deadlock will not kill tests.
	s.ctx, _ = context.WithTimeout(context.TODO(), 2*time.Second)
}

func (s *ClientInterceptorTestSuite) TearDownSuite() {
	if s.serverListener != nil {
		s.server.Stop()
		s.T().Logf("stopped grpc.Server at: %v", s.serverListener.Addr().String())
		s.serverListener.Close()

	}
	if s.clientConn != nil {
		s.clientConn.Close()
	}
}

func (s *ClientInterceptorTestSuite) TestUnaryIncrementsStarted() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingEmpty", "unary")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingEmpty", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_started_total should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingError", "unary")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.Unavailable)})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingError", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_started_total should be incremented for PingError")
}

func (s *ClientInterceptorTestSuite) TestUnaryIncrementsHandled() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingEmpty", "unary", "OK")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{}) // should return with code=OK
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingEmpty", "unary", "OK")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handled_count should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingError", "unary", "FailedPrecondition")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingError", "unary", "FailedPrecondition")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handled_total should be incremented for PingError")
}

func (s *ClientInterceptorTestSuite) TestUnaryIncrementsHistograms() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingEmpty", "unary")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{}) // should return with code=OK
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingEmpty", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handled_count should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingError", "unary")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingError", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handling_seconds_count should be incremented for PingError")
}

func (s *ClientInterceptorTestSuite) TestStreamingIncrementsStarted() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingList", "server_stream")
	s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_started_total", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_started_total should be incremented for PingList")
}

func (s *ClientInterceptorTestSuite) TestStreamingIncrementsHistograms() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingList", "server_stream")
	ss, _ := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{}) // should return with code=OK
	// Do a read, just for kicks.
	for {
		_, err := ss.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(s.T(), err, "reading pingList shouldn't fail")
	}
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handling_seconds_count should be incremented for PingList OK")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingList", "server_stream")
	ss, err := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	require.NoError(s.T(), err, "PingList must not fail immedietely")
	// Do a read, just to progate errors.
	_, err = ss.Recv()
	require.Equal(s.T(), codes.FailedPrecondition, grpc.Code(err), "Recv must return FailedPrecondition, otherwise the test is wrong")

	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handling_seconds_count", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handling_seconds_count should be incremented for PingList FailedPrecondition")
}

func (s *ClientInterceptorTestSuite) TestStreamingIncrementsHandled() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingList", "server_stream", "OK")
	ss, _ := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{}) // should return with code=OK
	// Do a read, just for kicks.
	for {
		_, err := ss.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(s.T(), err, "reading pingList shouldn't fail")
	}
	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingList", "server_stream", "OK")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handled_total should be incremented for PingList OK")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingList", "server_stream", "FailedPrecondition")
	ss, err := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	require.NoError(s.T(), err, "PingList must not fail immedietely")
	// Do a read, just to progate errors.
	_, err = ss.Recv()
	require.Equal(s.T(), codes.FailedPrecondition, grpc.Code(err), "Recv must return FailedPrecondition, otherwise the test is wrong")

	after = sumCountersForMetricAndLabels(s.T(), "grpc_client_handled_total", "PingList", "server_stream", "FailedPrecondition")
	assert.EqualValues(s.T(), before+1, after, "grpc_client_handled_total should be incremented for PingList FailedPrecondition")
}

func (s *ClientInterceptorTestSuite) TestStreamingIncrementsMessageCounts() {
	beforeRecv := sumCountersForMetricAndLabels(s.T(), "grpc_client_msg_received_total", "PingList", "server_stream")
	beforeSent := sumCountersForMetricAndLabels(s.T(), "grpc_client_msg_sent_total", "PingList", "server_stream")
	ss, _ := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{}) // should return with code=OK
	// Do a read, just for kicks.
	count := 0
	for {
		_, err := ss.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(s.T(), err, "reading pingList shouldn't fail")
		count += 1
	}
	require.EqualValues(s.T(), countListResponses, count, "Number of received msg on the wire must match")
	afterSent := sumCountersForMetricAndLabels(s.T(), "grpc_client_msg_sent_total", "PingList", "server_stream")
	afterRecv := sumCountersForMetricAndLabels(s.T(), "grpc_client_msg_received_total", "PingList", "server_stream")

	assert.EqualValues(s.T(), beforeSent+1, afterSent, "grpc_client_msg_sent_total should be incremented 20 times for PingList")
	assert.EqualValues(s.T(), beforeRecv+countListResponses, afterRecv, "grpc_client_msg_sent_total should be incremented ones for PingList ")
}
