package grpc_prometheus

import (
	"io"

	prom "github.com/prometheus/client_golang/prometheus"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ClientMetrics represents a collection of metrics to be registered on a
// Prometheus metrics registry for a gRPC client.
type ClientMetrics struct {
	clientStartedCounter          *prom.CounterVec
	clientHandledCounter          *prom.CounterVec
	clientStreamMsgReceived       *prom.CounterVec
	clientStreamMsgSent           *prom.CounterVec
	clientHandledHistogramEnabled bool
	clientHandledHistogramOpts    prom.HistogramOpts
	clientHandledHistogram        *prom.HistogramVec
}

// NewClientMetrics returns a ClientMetrics object. Use a new instance of
// ClientMetrics when not using the default Prometheus metrics registry, for
// example when wanting to control which metrics are added to a registry as
// opposed to automatically adding metrics via init functions.
func NewClientMetrics(counterOpts ...CounterOption) *ClientMetrics {
	opts := counterOptions(counterOpts)
	return &ClientMetrics{
		clientStartedCounter: prom.NewCounterVec(
			opts.apply(prom.CounterOpts{
				Name: "grpc_client_started_total",
				Help: "Total number of RPCs started on the client.",
			}), []string{"grpc_type", "grpc_service", "grpc_method"}),

		clientHandledCounter: prom.NewCounterVec(
			opts.apply(prom.CounterOpts{
				Name: "grpc_client_handled_total",
				Help: "Total number of RPCs completed by the client, regardless of success or failure.",
			}), []string{"grpc_type", "grpc_service", "grpc_method", "grpc_code"}),

		clientStreamMsgReceived: prom.NewCounterVec(
			opts.apply(prom.CounterOpts{
				Name: "grpc_client_msg_received_total",
				Help: "Total number of RPC stream messages received by the client.",
			}), []string{"grpc_type", "grpc_service", "grpc_method"}),

		clientStreamMsgSent: prom.NewCounterVec(
			opts.apply(prom.CounterOpts{
				Name: "grpc_client_msg_sent_total",
				Help: "Total number of gRPC stream messages sent by the client.",
			}), []string{"grpc_type", "grpc_service", "grpc_method"}),

		clientHandledHistogramEnabled: false,
		clientHandledHistogramOpts: prom.HistogramOpts{
			Name:    "grpc_client_handling_seconds",
			Help:    "Histogram of response latency (seconds) of the gRPC until it is finished by the application.",
			Buckets: prom.DefBuckets,
		},
		clientHandledHistogram: nil,
	}
}

// Describe sends the super-set of all possible descriptors of metrics
// collected by this Collector to the provided channel and returns once
// the last descriptor has been sent.
func (m *ClientMetrics) Describe(ch chan<- *prom.Desc) {
	m.clientStartedCounter.Describe(ch)
	m.clientHandledCounter.Describe(ch)
	m.clientStreamMsgReceived.Describe(ch)
	m.clientStreamMsgSent.Describe(ch)
	if m.clientHandledHistogramEnabled {
		m.clientHandledHistogram.Describe(ch)
	}
}

// Collect is called by the Prometheus registry when collecting
// metrics. The implementation sends each collected metric via the
// provided channel and returns once the last metric has been sent.
func (m *ClientMetrics) Collect(ch chan<- prom.Metric) {
	m.clientStartedCounter.Collect(ch)
	m.clientHandledCounter.Collect(ch)
	m.clientStreamMsgReceived.Collect(ch)
	m.clientStreamMsgSent.Collect(ch)
	if m.clientHandledHistogramEnabled {
		m.clientHandledHistogram.Collect(ch)
	}
}

// EnableClientHandlingTimeHistogram turns on recording of handling time of RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func (m *ClientMetrics) EnableClientHandlingTimeHistogram(opts ...HistogramOption) {
	for _, o := range opts {
		o(&m.clientHandledHistogramOpts)
	}
	if !m.clientHandledHistogramEnabled {
		m.clientHandledHistogram = prom.NewHistogramVec(
			m.clientHandledHistogramOpts,
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
	}
	m.clientHandledHistogramEnabled = true
}

// UnaryClientInterceptor is a gRPC client-side interceptor that provides Prometheus monitoring for Unary RPCs.
func (m *ClientMetrics) UnaryClientInterceptor() func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		monitor := newClientReporter(m, Unary, method)
		monitor.SentMessage()
		err := invoker(ctx, method, req, reply, cc, opts...)
		if err != nil {
			monitor.ReceivedMessage()
		}
		st, _ := status.FromError(err)
		monitor.Handled(st.Code())
		return err
	}
}

// StreamClientInterceptor is a gRPC client-side interceptor that provides Prometheus monitoring for Streaming RPCs.
func (m *ClientMetrics) StreamClientInterceptor() func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		monitor := newClientReporter(m, clientStreamType(desc), method)
		clientStream, err := streamer(ctx, desc, cc, method, opts...)
		if err != nil {
			st, _ := status.FromError(err)
			monitor.Handled(st.Code())
			return nil, err
		}
		return &monitoredClientStream{clientStream, monitor}, nil
	}
}

func clientStreamType(desc *grpc.StreamDesc) grpcType {
	if desc.ClientStreams && !desc.ServerStreams {
		return ClientStream
	} else if !desc.ClientStreams && desc.ServerStreams {
		return ServerStream
	}
	return BidiStream
}

// monitoredClientStream wraps grpc.ClientStream allowing each Sent/Recv of message to increment counters.
type monitoredClientStream struct {
	grpc.ClientStream
	monitor *clientReporter
}

func (s *monitoredClientStream) SendMsg(m interface{}) error {
	err := s.ClientStream.SendMsg(m)
	if err == nil {
		s.monitor.SentMessage()
	}
	return err
}

func (s *monitoredClientStream) RecvMsg(m interface{}) error {
	err := s.ClientStream.RecvMsg(m)
	if err == nil {
		s.monitor.ReceivedMessage()
	} else if err == io.EOF {
		s.monitor.Handled(codes.OK)
	} else {
		st, _ := status.FromError(err)
		s.monitor.Handled(st.Code())
	}
	return err
}
