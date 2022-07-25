package oc

import (
	"github.com/sirupsen/logrus"
	"go.opencensus.io/trace"
)

var _ = (trace.Exporter)(&LogrusExporter{})

// LogrusExporter is an OpenCensus `trace.Exporter` that exports
// `trace.SpanData` to logrus output.
type LogrusExporter struct {
}

// ExportSpan exports `s` based on the the following rules:
//
// 1. All output will contain `s.Attributes`, `s.TraceID`, `s.SpanID`,
// `s.ParentSpanID` for correlation
//
// 2. Any calls to .Annotate will not be supported.
//
// 3. The span itself will be written at `logrus.InfoLevel` unless
// `s.Status.Code != 0` in which case it will be written at `logrus.ErrorLevel`
// providing `s.Status.Message` as the error value.
func (le *LogrusExporter) ExportSpan(s *trace.SpanData) {
	// Combine all span annotations with traceID, spanID, parentSpanID
	baseEntry := logrus.WithFields(logrus.Fields(s.Attributes))
	baseEntry.Data["traceID"] = s.TraceID.String()
	baseEntry.Data["spanID"] = s.SpanID.String()
	baseEntry.Data["parentSpanID"] = s.ParentSpanID.String()
	baseEntry.Data["startTime"] = s.StartTime
	baseEntry.Data["endTime"] = s.EndTime
	baseEntry.Data["duration"] = s.EndTime.Sub(s.StartTime).String()
	baseEntry.Data["name"] = s.Name
	baseEntry.Time = s.StartTime

	level := logrus.InfoLevel
	if s.Status.Code != 0 {
		level = logrus.ErrorLevel
		baseEntry.Data[logrus.ErrorKey] = s.Status.Message
	}
	baseEntry.Log(level, "Span")
}
