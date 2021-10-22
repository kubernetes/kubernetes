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
//

package metricexport

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricproducer"
	"go.opencensus.io/trace"
)

var (
	defaultSampler             = trace.ProbabilitySampler(0.0001)
	errReportingIntervalTooLow = fmt.Errorf("reporting interval less than %d", minimumReportingDuration)
	errAlreadyStarted          = fmt.Errorf("already started")
	errIntervalReaderNil       = fmt.Errorf("interval reader is nil")
	errExporterNil             = fmt.Errorf("exporter is nil")
	errReaderNil               = fmt.Errorf("reader is nil")
)

const (
	defaultReportingDuration = 60 * time.Second
	minimumReportingDuration = 1 * time.Second
	defaultSpanName          = "ExportMetrics"
)

// ReaderOptions contains options pertaining to metrics reader.
type ReaderOptions struct {
	// SpanName is the name used for span created to export metrics.
	SpanName string
}

// Reader reads metrics from all producers registered
// with producer manager and exports those metrics using provided
// exporter.
type Reader struct {
	sampler trace.Sampler

	spanName string
}

// IntervalReader periodically reads metrics from all producers registered
// with producer manager and exports those metrics using provided
// exporter. Call Reader.Stop() to stop the reader.
type IntervalReader struct {
	// ReportingInterval it the time duration between two consecutive
	// metrics reporting. defaultReportingDuration  is used if it is not set.
	// It cannot be set lower than minimumReportingDuration.
	ReportingInterval time.Duration

	exporter   Exporter
	timer      *time.Ticker
	quit, done chan bool
	mu         sync.RWMutex
	reader     *Reader
}

// ReaderOption apply changes to ReaderOptions.
type ReaderOption func(*ReaderOptions)

// WithSpanName makes new reader to use given span name when exporting metrics.
func WithSpanName(spanName string) ReaderOption {
	return func(o *ReaderOptions) {
		o.SpanName = spanName
	}
}

// NewReader returns a reader configured with specified options.
func NewReader(o ...ReaderOption) *Reader {
	var opts ReaderOptions
	for _, op := range o {
		op(&opts)
	}
	reader := &Reader{defaultSampler, defaultSpanName}
	if opts.SpanName != "" {
		reader.spanName = opts.SpanName
	}
	return reader
}

// NewIntervalReader creates a reader. Once started it periodically
// reads metrics from all producers and exports them using provided exporter.
func NewIntervalReader(reader *Reader, exporter Exporter) (*IntervalReader, error) {
	if exporter == nil {
		return nil, errExporterNil
	}
	if reader == nil {
		return nil, errReaderNil
	}

	r := &IntervalReader{
		exporter: exporter,
		reader:   reader,
	}
	return r, nil
}

// Start starts the IntervalReader which periodically reads metrics from all
// producers registered with global producer manager. If the reporting interval
// is not set prior to calling this function then default reporting interval
// is used.
func (ir *IntervalReader) Start() error {
	if ir == nil {
		return errIntervalReaderNil
	}
	ir.mu.Lock()
	defer ir.mu.Unlock()
	var reportingInterval = defaultReportingDuration
	if ir.ReportingInterval != 0 {
		if ir.ReportingInterval < minimumReportingDuration {
			return errReportingIntervalTooLow
		}
		reportingInterval = ir.ReportingInterval
	}

	if ir.done != nil {
		return errAlreadyStarted
	}
	ir.timer = time.NewTicker(reportingInterval)
	ir.quit = make(chan bool)
	ir.done = make(chan bool)

	go ir.startInternal()
	return nil
}

func (ir *IntervalReader) startInternal() {
	for {
		select {
		case <-ir.timer.C:
			ir.reader.ReadAndExport(ir.exporter)
		case <-ir.quit:
			ir.timer.Stop()
			ir.done <- true
			return
		}
	}
}

// Stop stops the reader from reading and exporting metrics.
// Additional call to Stop are no-ops.
func (ir *IntervalReader) Stop() {
	if ir == nil {
		return
	}
	ir.mu.Lock()
	defer ir.mu.Unlock()
	if ir.quit == nil {
		return
	}
	ir.quit <- true
	<-ir.done
	close(ir.quit)
	close(ir.done)
	ir.quit = nil
}

// ReadAndExport reads metrics from all producer registered with
// producer manager and then exports them using provided exporter.
func (r *Reader) ReadAndExport(exporter Exporter) {
	ctx, span := trace.StartSpan(context.Background(), r.spanName, trace.WithSampler(r.sampler))
	defer span.End()
	producers := metricproducer.GlobalManager().GetAll()
	data := []*metricdata.Metric{}
	for _, producer := range producers {
		data = append(data, producer.Read()...)
	}
	// TODO: [rghetia] add metrics for errors.
	exporter.ExportMetrics(ctx, data)
}
