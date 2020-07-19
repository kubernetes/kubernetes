package log

import (
	"context"

	"github.com/sirupsen/logrus"
	"go.opencensus.io/trace"
)

// G returns a `logrus.Entry` with the `TraceID, SpanID` from `ctx` if `ctx`
// contains an OpenCensus `trace.Span`.
func G(ctx context.Context) *logrus.Entry {
	span := trace.FromContext(ctx)
	if span != nil {
		sctx := span.SpanContext()
		return logrus.WithFields(logrus.Fields{
			"traceID": sctx.TraceID.String(),
			"spanID":  sctx.SpanID.String(),
			// "parentSpanID": TODO: JTERRY75 - Try to convince OC to export this?
		})
	}
	return logrus.NewEntry(logrus.StandardLogger())
}
