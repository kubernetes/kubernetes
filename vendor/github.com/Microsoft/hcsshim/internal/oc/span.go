package oc

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/log"
	"go.opencensus.io/trace"
)

var DefaultSampler = trace.AlwaysSample()

// SetSpanStatus sets `span.SetStatus` to the proper status depending on `err`. If
// `err` is `nil` assumes `trace.StatusCodeOk`.
func SetSpanStatus(span *trace.Span, err error) {
	status := trace.Status{}
	if err != nil {
		status.Code = int32(toStatusCode(err))
		status.Message = err.Error()
	}
	span.SetStatus(status)
}

// StartSpan wraps "go.opencensus.io/trace".StartSpan, but, if the span is sampling,
// adds a log entry to the context that points to the newly created span.
func StartSpan(ctx context.Context, name string, o ...trace.StartOption) (context.Context, *trace.Span) {
	ctx, s := trace.StartSpan(ctx, name, o...)
	return update(ctx, s)
}

// StartSpanWithRemoteParent wraps "go.opencensus.io/trace".StartSpanWithRemoteParent.
//
// See StartSpan for more information.
func StartSpanWithRemoteParent(ctx context.Context, name string, parent trace.SpanContext, o ...trace.StartOption) (context.Context, *trace.Span) {
	ctx, s := trace.StartSpanWithRemoteParent(ctx, name, parent, o...)
	return update(ctx, s)
}

func update(ctx context.Context, s *trace.Span) (context.Context, *trace.Span) {
	if s.IsRecordingEvents() {
		ctx = log.UpdateContext(ctx)
	}

	return ctx, s
}

var WithServerSpanKind = trace.WithSpanKind(trace.SpanKindServer)
var WithClientSpanKind = trace.WithSpanKind(trace.SpanKindClient)

func spanKindToString(sk int) string {
	switch sk {
	case trace.SpanKindClient:
		return "client"
	case trace.SpanKindServer:
		return "server"
	default:
		return ""
	}
}
