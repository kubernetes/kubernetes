// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package rpcmetrics

import (
	"strconv"
	"sync"
	"time"

	"github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	"github.com/uber/jaeger-lib/metrics"

	jaeger "github.com/uber/jaeger-client-go"
)

const defaultMaxNumberOfOperations = 200

// Observer is an observer that can emit RPC metrics.
type Observer struct {
	metricsByOperation *MetricsByOperation
}

// NewObserver creates a new observer that can emit RPC metrics.
func NewObserver(metricsFactory metrics.Factory, normalizer NameNormalizer) *Observer {
	return &Observer{
		metricsByOperation: newMetricsByOperation(
			metricsFactory,
			normalizer,
			defaultMaxNumberOfOperations,
		),
	}
}

// OnStartSpan creates a new Observer for the span.
func (o *Observer) OnStartSpan(
	operationName string,
	options opentracing.StartSpanOptions,
) jaeger.SpanObserver {
	return NewSpanObserver(o.metricsByOperation, operationName, options)
}

// SpanKind identifies the span as inboud, outbound, or internal
type SpanKind int

const (
	// Local span kind
	Local SpanKind = iota
	// Inbound span kind
	Inbound
	// Outbound span kind
	Outbound
)

// SpanObserver collects RPC metrics
type SpanObserver struct {
	metricsByOperation *MetricsByOperation
	operationName      string
	startTime          time.Time
	mux                sync.Mutex
	kind               SpanKind
	httpStatusCode     uint16
	err                bool
}

// NewSpanObserver creates a new SpanObserver that can emit RPC metrics.
func NewSpanObserver(
	metricsByOperation *MetricsByOperation,
	operationName string,
	options opentracing.StartSpanOptions,
) *SpanObserver {
	so := &SpanObserver{
		metricsByOperation: metricsByOperation,
		operationName:      operationName,
		startTime:          options.StartTime,
	}
	for k, v := range options.Tags {
		so.handleTagInLock(k, v)
	}
	return so
}

// handleTags watches for special tags
// - SpanKind
// - HttpStatusCode
// - Error
func (so *SpanObserver) handleTagInLock(key string, value interface{}) {
	if key == string(ext.SpanKind) {
		if v, ok := value.(ext.SpanKindEnum); ok {
			value = string(v)
		}
		if v, ok := value.(string); ok {
			if v == string(ext.SpanKindRPCClientEnum) {
				so.kind = Outbound
			} else if v == string(ext.SpanKindRPCServerEnum) {
				so.kind = Inbound
			}
		}
		return
	}
	if key == string(ext.HTTPStatusCode) {
		if v, ok := value.(uint16); ok {
			so.httpStatusCode = v
		} else if v, ok := value.(int); ok {
			so.httpStatusCode = uint16(v)
		} else if v, ok := value.(string); ok {
			if vv, err := strconv.Atoi(v); err == nil {
				so.httpStatusCode = uint16(vv)
			}
		}
		return
	}
	if key == string(ext.Error) {
		if v, ok := value.(bool); ok {
			so.err = v
		} else if v, ok := value.(string); ok {
			if vv, err := strconv.ParseBool(v); err == nil {
				so.err = vv
			}
		}
		return
	}
}

// OnFinish emits the RPC metrics. It only has an effect when operation name
// is not blank, and the span kind is an RPC server.
func (so *SpanObserver) OnFinish(options opentracing.FinishOptions) {
	so.mux.Lock()
	defer so.mux.Unlock()

	if so.operationName == "" || so.kind != Inbound {
		return
	}

	mets := so.metricsByOperation.get(so.operationName)
	latency := options.FinishTime.Sub(so.startTime)
	if so.err {
		mets.RequestCountFailures.Inc(1)
		mets.RequestLatencyFailures.Record(latency)
	} else {
		mets.RequestCountSuccess.Inc(1)
		mets.RequestLatencySuccess.Record(latency)
	}
	mets.recordHTTPStatusCode(so.httpStatusCode)
}

// OnSetOperationName records new operation name.
func (so *SpanObserver) OnSetOperationName(operationName string) {
	so.mux.Lock()
	so.operationName = operationName
	so.mux.Unlock()
}

// OnSetTag implements SpanObserver
func (so *SpanObserver) OnSetTag(key string, value interface{}) {
	so.mux.Lock()
	so.handleTagInLock(key, value)
	so.mux.Unlock()
}
