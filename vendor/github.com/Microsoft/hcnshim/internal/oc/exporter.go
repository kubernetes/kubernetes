package oc

import (
	"github.com/sirupsen/logrus"
	"go.opencensus.io/trace"
	"google.golang.org/grpc/codes"

	"github.com/Microsoft/hcnshim/internal/log"
	"github.com/Microsoft/hcnshim/internal/logfields"
)

const spanMessage = "Span"

var _errorCodeKey = logrus.ErrorKey + "Code"

// LogrusExporter is an OpenCensus `trace.Exporter` that exports
// `trace.SpanData` to logrus output.
type LogrusExporter struct{}

var _ trace.Exporter = &LogrusExporter{}

// ExportSpan exports `s` based on the the following rules:
//
// 1. All output will contain `s.Attributes`, `s.SpanKind`, `s.TraceID`,
// `s.SpanID`, and `s.ParentSpanID` for correlation
//
// 2. Any calls to .Annotate will not be supported.
//
// 3. The span itself will be written at `logrus.InfoLevel` unless
// `s.Status.Code != 0` in which case it will be written at `logrus.ErrorLevel`
// providing `s.Status.Message` as the error value.
func (le *LogrusExporter) ExportSpan(s *trace.SpanData) {
	if s.DroppedAnnotationCount > 0 {
		logrus.WithFields(logrus.Fields{
			"name":            s.Name,
			logfields.TraceID: s.TraceID.String(),
			logfields.SpanID:  s.SpanID.String(),
			"dropped":         s.DroppedAttributeCount,
			"maxAttributes":   len(s.Attributes),
		}).Warning("span had dropped attributes")
	}

	entry := log.L.Dup()
	// Combine all span annotations with span data (eg, trace ID, span ID, parent span ID,
	// error, status code)
	// (OC) Span attributes are guaranteed to be  strings, bools, or int64s, so we can
	// can skip overhead in entry.WithFields() and add them directly to entry.Data.
	// Preallocate ahead of time, since we should add, at most, 10 additional entries
	data := make(logrus.Fields, len(entry.Data)+len(s.Attributes)+10)

	// Default log entry may have prexisting/application-wide data
	for k, v := range entry.Data {
		data[k] = v
	}
	for k, v := range s.Attributes {
		data[k] = v
	}

	data[logfields.Name] = s.Name
	data[logfields.TraceID] = s.TraceID.String()
	data[logfields.SpanID] = s.SpanID.String()
	data[logfields.ParentSpanID] = s.ParentSpanID.String()
	data[logfields.StartTime] = s.StartTime
	data[logfields.EndTime] = s.EndTime
	data[logfields.Duration] = s.EndTime.Sub(s.StartTime)
	if sk := spanKindToString(s.SpanKind); sk != "" {
		data["spanKind"] = sk
	}

	level := logrus.InfoLevel
	if s.Status.Code != 0 {
		level = logrus.ErrorLevel

		// don't overwrite an existing "error" or "errorCode" attributes
		if _, ok := data[logrus.ErrorKey]; !ok {
			data[logrus.ErrorKey] = s.Status.Message
		}
		if _, ok := data[_errorCodeKey]; !ok {
			data[_errorCodeKey] = codes.Code(s.Status.Code).String()
		}
	}

	entry.Data = data
	entry.Time = s.StartTime
	entry.Log(level, spanMessage)
}
