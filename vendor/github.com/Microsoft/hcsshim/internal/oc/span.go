package oc

import (
	"go.opencensus.io/trace"
)

// SetSpanStatus sets `span.SetStatus` to the proper status depending on `err`. If
// `err` is `nil` assumes `trace.StatusCodeOk`.
func SetSpanStatus(span *trace.Span, err error) {
	status := trace.Status{}
	if err != nil {
		// TODO: JTERRY75 - Handle errors in a non-generic way
		status.Code = trace.StatusCodeUnknown
		status.Message = err.Error()
	}
	span.SetStatus(status)
}
