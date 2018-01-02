// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

package grpc_prometheus

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	pb_testproto "github.com/grpc-ecosystem/go-grpc-prometheus/examples/testproto"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

const (
	pingDefaultValue   = "I like kittens."
	countListResponses = 20
)

func TestServerInterceptorSuite(t *testing.T) {
	suite.Run(t, &ServerInterceptorTestSuite{})
}

type ServerInterceptorTestSuite struct {
	suite.Suite

	serverListener net.Listener
	server         *grpc.Server
	clientConn     *grpc.ClientConn
	testClient     pb_testproto.TestServiceClient
	ctx            context.Context
}

func (s *ServerInterceptorTestSuite) SetupSuite() {
	var err error

	EnableHandlingTimeHistogram()

	s.serverListener, err = net.Listen("tcp", "127.0.0.1:0")
	require.NoError(s.T(), err, "must be able to allocate a port for serverListener")

	// This is the point where we hook up the interceptor
	s.server = grpc.NewServer(
		grpc.StreamInterceptor(StreamServerInterceptor),
		grpc.UnaryInterceptor(UnaryServerInterceptor),
	)
	pb_testproto.RegisterTestServiceServer(s.server, &testService{t: s.T()})

	go func() {
		s.server.Serve(s.serverListener)
	}()

	s.clientConn, err = grpc.Dial(s.serverListener.Addr().String(), grpc.WithInsecure(), grpc.WithBlock(), grpc.WithTimeout(2*time.Second))
	require.NoError(s.T(), err, "must not error on client Dial")
	s.testClient = pb_testproto.NewTestServiceClient(s.clientConn)

	// Important! Pre-register stuff here.
	Register(s.server)
}

func (s *ServerInterceptorTestSuite) SetupTest() {
	// Make all RPC calls last at most 2 sec, meaning all async issues or deadlock will not kill tests.
	s.ctx, _ = context.WithTimeout(context.TODO(), 2*time.Second)
}

func (s *ServerInterceptorTestSuite) TearDownSuite() {
	if s.serverListener != nil {
		s.server.Stop()
		s.T().Logf("stopped grpc.Server at: %v", s.serverListener.Addr().String())
		s.serverListener.Close()

	}
	if s.clientConn != nil {
		s.clientConn.Close()
	}
}

func (s *ServerInterceptorTestSuite) TestRegisterPresetsStuff() {
	for testId, testCase := range []struct {
		metricName     string
		existingLabels []string
	}{
		{"grpc_server_started_total", []string{"mwitkow.testproto.TestService", "PingEmpty", "unary"}},
		{"grpc_server_started_total", []string{"mwitkow.testproto.TestService", "PingList", "server_stream"}},
		{"grpc_server_msg_received_total", []string{"mwitkow.testproto.TestService", "PingList", "server_stream"}},
		{"grpc_server_msg_sent_total", []string{"mwitkow.testproto.TestService", "PingEmpty", "unary"}},
		{"grpc_server_handling_seconds_sum", []string{"mwitkow.testproto.TestService", "PingEmpty", "unary"}},
		{"grpc_server_handling_seconds_count", []string{"mwitkow.testproto.TestService", "PingList", "server_stream"}},
		{"grpc_server_handled_total", []string{"mwitkow.testproto.TestService", "PingList", "server_stream", "OutOfRange"}},
		{"grpc_server_handled_total", []string{"mwitkow.testproto.TestService", "PingList", "server_stream", "Aborted"}},
		{"grpc_server_handled_total", []string{"mwitkow.testproto.TestService", "PingEmpty", "unary", "FailedPrecondition"}},
		{"grpc_server_handled_total", []string{"mwitkow.testproto.TestService", "PingEmpty", "unary", "ResourceExhausted"}},
	} {
		lineCount := len(fetchPrometheusLines(s.T(), testCase.metricName, testCase.existingLabels...))
		assert.NotEqual(s.T(), 0, lineCount, "metrics must exist for test case %d", testId)
	}
}

func (s *ServerInterceptorTestSuite) TestUnaryIncrementsStarted() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingEmpty", "unary")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingEmpty", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_started_total should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingError", "unary")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.Unavailable)})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingError", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_started_total should be incremented for PingError")
}

func (s *ServerInterceptorTestSuite) TestUnaryIncrementsHandled() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingEmpty", "unary", "OK")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{}) // should return with code=OK
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingEmpty", "unary", "OK")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handled_count should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingError", "unary", "FailedPrecondition")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingError", "unary", "FailedPrecondition")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handled_total should be incremented for PingError")
}

func (s *ServerInterceptorTestSuite) TestUnaryIncrementsHistograms() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingEmpty", "unary")
	s.testClient.PingEmpty(s.ctx, &pb_testproto.Empty{}) // should return with code=OK
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingEmpty", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handled_count should be incremented for PingEmpty")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingError", "unary")
	s.testClient.PingError(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingError", "unary")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handling_seconds_count should be incremented for PingError")
}

func (s *ServerInterceptorTestSuite) TestStreamingIncrementsStarted() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingList", "server_stream")
	s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{})
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_started_total", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_started_total should be incremented for PingList")
}

func (s *ServerInterceptorTestSuite) TestStreamingIncrementsHistograms() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingList", "server_stream")
	ss, _ := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{}) // should return with code=OK
	// Do a read, just for kicks.
	for {
		_, err := ss.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(s.T(), err, "reading pingList shouldn't fail")
	}
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handling_seconds_count should be incremented for PingList OK")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingList", "server_stream")
	_, err := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	require.NoError(s.T(), err, "PingList must not fail immedietely")

	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handling_seconds_count", "PingList", "server_stream")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handling_seconds_count should be incremented for PingList FailedPrecondition")
}

func (s *ServerInterceptorTestSuite) TestStreamingIncrementsHandled() {
	var before int
	var after int

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingList", "server_stream", "OK")
	ss, _ := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{}) // should return with code=OK
	// Do a read, just for kicks.
	for {
		_, err := ss.Recv()
		if err == io.EOF {
			break
		}
		require.NoError(s.T(), err, "reading pingList shouldn't fail")
	}
	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingList", "server_stream", "OK")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handled_total should be incremented for PingList OK")

	before = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingList", "server_stream", "FailedPrecondition")
	_, err := s.testClient.PingList(s.ctx, &pb_testproto.PingRequest{ErrorCodeReturned: uint32(codes.FailedPrecondition)}) // should return with code=FailedPrecondition
	require.NoError(s.T(), err, "PingList must not fail immedietely")

	after = sumCountersForMetricAndLabels(s.T(), "grpc_server_handled_total", "PingList", "server_stream", "FailedPrecondition")
	assert.EqualValues(s.T(), before+1, after, "grpc_server_handled_total should be incremented for PingList FailedPrecondition")
}

func (s *ServerInterceptorTestSuite) TestStreamingIncrementsMessageCounts() {
	beforeRecv := sumCountersForMetricAndLabels(s.T(), "grpc_server_msg_received_total", "PingList", "server_stream")
	beforeSent := sumCountersForMetricAndLabels(s.T(), "grpc_server_msg_sent_total", "PingList", "server_stream")
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
	afterSent := sumCountersForMetricAndLabels(s.T(), "grpc_server_msg_sent_total", "PingList", "server_stream")
	afterRecv := sumCountersForMetricAndLabels(s.T(), "grpc_server_msg_received_total", "PingList", "server_stream")

	assert.EqualValues(s.T(), beforeSent+countListResponses, afterSent, "grpc_server_msg_sent_total should be incremented 20 times for PingList")
	assert.EqualValues(s.T(), beforeRecv+1, afterRecv, "grpc_server_msg_sent_total should be incremented ones for PingList ")
}

func fetchPrometheusLines(t *testing.T, metricName string, matchingLabelValues ...string) []string {
	resp := httptest.NewRecorder()
	req, err := http.NewRequest("GET", "/", nil)
	require.NoError(t, err, "failed creating request for Prometheus handler")
	prometheus.Handler().ServeHTTP(resp, req)
	reader := bufio.NewReader(resp.Body)
	ret := []string{}
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		} else {
			require.NoError(t, err, "error reading stuff")
		}
		if !strings.HasPrefix(line, metricName) {
			continue
		}
		matches := true
		for _, labelValue := range matchingLabelValues {
			if !strings.Contains(line, `"`+labelValue+`"`) {
				matches = false
			}
		}
		if matches {
			ret = append(ret, line)
		}

	}
	return ret
}

func sumCountersForMetricAndLabels(t *testing.T, metricName string, matchingLabelValues ...string) int {
	count := 0
	for _, line := range fetchPrometheusLines(t, metricName, matchingLabelValues...) {
		valueString := line[strings.LastIndex(line, " ")+1 : len(line)-1]
		valueFloat, err := strconv.ParseFloat(valueString, 32)
		require.NoError(t, err, "failed parsing value for line: %v", line)
		count += int(valueFloat)
	}
	return count
}

type testService struct {
	t *testing.T
}

func (s *testService) PingEmpty(ctx context.Context, _ *pb_testproto.Empty) (*pb_testproto.PingResponse, error) {
	return &pb_testproto.PingResponse{Value: pingDefaultValue, Counter: 42}, nil
}

func (s *testService) Ping(ctx context.Context, ping *pb_testproto.PingRequest) (*pb_testproto.PingResponse, error) {
	// Send user trailers and headers.
	return &pb_testproto.PingResponse{Value: ping.Value, Counter: 42}, nil
}

func (s *testService) PingError(ctx context.Context, ping *pb_testproto.PingRequest) (*pb_testproto.Empty, error) {
	code := codes.Code(ping.ErrorCodeReturned)
	return nil, grpc.Errorf(code, "Userspace error.")
}

func (s *testService) PingList(ping *pb_testproto.PingRequest, stream pb_testproto.TestService_PingListServer) error {
	if ping.ErrorCodeReturned != 0 {
		return grpc.Errorf(codes.Code(ping.ErrorCodeReturned), "foobar")
	}
	// Send user trailers and headers.
	for i := 0; i < countListResponses; i++ {
		stream.Send(&pb_testproto.PingResponse{Value: ping.Value, Counter: int32(i)})
	}
	return nil
}
