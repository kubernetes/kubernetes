// Copyright 2019, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package exporter contains a log exporter that supports exporting
// OpenCensus metrics and spans to a logging framework.
package exporter // import "go.opencensus.io/examples/exporter"

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricexport"
	"go.opencensus.io/trace"
)

// LogExporter exports metrics and span to log file
type LogExporter struct {
	reader         *metricexport.Reader
	ir             *metricexport.IntervalReader
	initReaderOnce sync.Once
	o              Options
	tFile          *os.File
	mFile          *os.File
	tLogger        *log.Logger
	mLogger        *log.Logger
}

// Options provides options for LogExporter
type Options struct {
	// ReportingInterval is a time interval between two successive metrics
	// export.
	ReportingInterval time.Duration

	// MetricsLogFile is path where exported metrics are logged.
	// If it is nil then the metrics are logged on console
	MetricsLogFile string

	// TracesLogFile is path where exported span data are logged.
	// If it is nil then the span data are logged on console
	TracesLogFile string
}

func getLogger(filepath string) (*log.Logger, *os.File, error) {
	if filepath == "" {
		return log.New(os.Stdout, "", 0), nil, nil
	}
	f, err := os.OpenFile(filepath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return nil, nil, err
	}
	return log.New(f, "", 0), f, nil
}

// NewLogExporter creates new log exporter.
func NewLogExporter(options Options) (*LogExporter, error) {
	e := &LogExporter{reader: metricexport.NewReader(),
		o: options}
	var err error
	e.tLogger, e.tFile, err = getLogger(options.TracesLogFile)
	if err != nil {
		return nil, err
	}
	e.mLogger, e.mFile, err = getLogger(options.MetricsLogFile)
	if err != nil {
		return nil, err
	}
	return e, nil
}

func printMetricDescriptor(metric *metricdata.Metric) string {
	d := metric.Descriptor
	return fmt.Sprintf("name: %s, type: %s, unit: %s ",
		d.Name, d.Type, d.Unit)
}

func printLabels(metric *metricdata.Metric, values []metricdata.LabelValue) string {
	d := metric.Descriptor
	kv := []string{}
	for i, k := range d.LabelKeys {
		kv = append(kv, fmt.Sprintf("%s=%v", k, values[i]))
	}
	return fmt.Sprintf("%v", kv)
}

func printPoint(point metricdata.Point) string {
	switch v := point.Value.(type) {
	case *metricdata.Distribution:
		dv := v
		return fmt.Sprintf("count=%v sum=%v sum_sq_dev=%v, buckets=%v", dv.Count,
			dv.Sum, dv.SumOfSquaredDeviation, dv.Buckets)
	default:
		return fmt.Sprintf("value=%v", point.Value)
	}
}

// Start starts the metric and span data exporter.
func (e *LogExporter) Start() error {
	trace.RegisterExporter(e)
	e.initReaderOnce.Do(func() {
		e.ir, _ = metricexport.NewIntervalReader(&metricexport.Reader{}, e)
	})
	e.ir.ReportingInterval = e.o.ReportingInterval
	return e.ir.Start()
}

// Stop stops the metric and span data exporter.
func (e *LogExporter) Stop() {
	trace.UnregisterExporter(e)
	e.ir.Stop()
}

// Close closes any files that were opened for logging.
func (e *LogExporter) Close() {
	if e.tFile != nil {
		e.tFile.Close()
		e.tFile = nil
	}
	if e.mFile != nil {
		e.mFile.Close()
		e.mFile = nil
	}
}

// ExportMetrics exports to log.
func (e *LogExporter) ExportMetrics(ctx context.Context, metrics []*metricdata.Metric) error {
	for _, metric := range metrics {
		for _, ts := range metric.TimeSeries {
			for _, point := range ts.Points {
				e.mLogger.Println("#----------------------------------------------")
				e.mLogger.Println()
				e.mLogger.Printf("Metric: %s\n  Labels: %s\n    Value : %s\n",
					printMetricDescriptor(metric),
					printLabels(metric, ts.LabelValues),
					printPoint(point))
				e.mLogger.Println()
			}
		}
	}
	return nil
}

// ExportSpan exports a SpanData to log
func (e *LogExporter) ExportSpan(sd *trace.SpanData) {
	var (
		traceID      = hex.EncodeToString(sd.SpanContext.TraceID[:])
		spanID       = hex.EncodeToString(sd.SpanContext.SpanID[:])
		parentSpanID = hex.EncodeToString(sd.ParentSpanID[:])
	)
	e.tLogger.Println()
	e.tLogger.Println("#----------------------------------------------")
	e.tLogger.Println()
	e.tLogger.Println("TraceID:     ", traceID)
	e.tLogger.Println("SpanID:      ", spanID)
	if !reZero.MatchString(parentSpanID) {
		e.tLogger.Println("ParentSpanID:", parentSpanID)
	}

	e.tLogger.Println()
	e.tLogger.Printf("Span:    %v\n", sd.Name)
	e.tLogger.Printf("Status:  %v [%v]\n", sd.Status.Message, sd.Status.Code)
	e.tLogger.Printf("Elapsed: %v\n", sd.EndTime.Sub(sd.StartTime).Round(time.Millisecond))

	spanKinds := map[int]string{
		1: "Server",
		2: "Client",
	}
	if spanKind, ok := spanKinds[sd.SpanKind]; ok {
		e.tLogger.Printf("SpanKind: %s\n", spanKind)
	}

	if len(sd.Annotations) > 0 {
		e.tLogger.Println()
		e.tLogger.Println("Annotations:")
		for _, item := range sd.Annotations {
			e.tLogger.Print(indent, item.Message)
			for k, v := range item.Attributes {
				e.tLogger.Printf(" %v=%v", k, v)
			}
			e.tLogger.Println()
		}
	}

	if len(sd.Attributes) > 0 {
		e.tLogger.Println()
		e.tLogger.Println("Attributes:")
		for k, v := range sd.Attributes {
			e.tLogger.Printf("%v- %v=%v\n", indent, k, v)
		}
	}

	if len(sd.MessageEvents) > 0 {
		eventTypes := map[trace.MessageEventType]string{
			trace.MessageEventTypeSent: "Sent",
			trace.MessageEventTypeRecv: "Received",
		}
		e.tLogger.Println()
		e.tLogger.Println("MessageEvents:")
		for _, item := range sd.MessageEvents {
			if eventType, ok := eventTypes[item.EventType]; ok {
				e.tLogger.Print(eventType)
			}
			e.tLogger.Printf("UncompressedByteSize: %v", item.UncompressedByteSize)
			e.tLogger.Printf("CompressedByteSize: %v", item.CompressedByteSize)
			e.tLogger.Println()
		}
	}
}
