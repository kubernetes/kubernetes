// Copyright The OpenTelemetry Authors
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

package trace

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	export "go.opentelemetry.io/otel/sdk/export/trace"
)

const (
	defaultMaxQueueSize       = 2048
	defaultScheduledDelay     = 5000 * time.Millisecond
	defaultMaxExportBatchSize = 512
)

var (
	errNilExporter = errors.New("exporter is nil")
)

type BatchSpanProcessorOption func(o *BatchSpanProcessorOptions)

type BatchSpanProcessorOptions struct {
	// MaxQueueSize is the maximum queue size to buffer spans for delayed processing. If the
	// queue gets full it drops the spans. Use BlockOnQueueFull to change this behavior.
	// The default value of MaxQueueSize is 2048.
	MaxQueueSize int

	// ScheduledDelayMillis is the delay interval in milliseconds between two consecutive
	// processing of batches.
	// The default value of ScheduledDelayMillis is 5000 msec.
	ScheduledDelayMillis time.Duration

	// MaxExportBatchSize is the maximum number of spans to process in a single batch.
	// If there are more than one batch worth of spans then it processes multiple batches
	// of spans one batch after the other without any delay.
	// The default value of MaxExportBatchSize is 512.
	MaxExportBatchSize int

	// BlockOnQueueFull blocks onEnd() and onStart() method if the queue is full
	// AND if BlockOnQueueFull is set to true.
	// Blocking option should be used carefully as it can severely affect the performance of an
	// application.
	BlockOnQueueFull bool
}

// BatchSpanProcessor implements SpanProcessor interfaces. It is used by
// exporters to receive export.SpanData asynchronously.
// Use BatchSpanProcessorOptions to change the behavior of the processor.
type BatchSpanProcessor struct {
	e export.SpanBatcher
	o BatchSpanProcessorOptions

	queue   chan *export.SpanData
	dropped uint32

	stopWait sync.WaitGroup
	stopOnce sync.Once
	stopCh   chan struct{}
}

var _ SpanProcessor = (*BatchSpanProcessor)(nil)

// NewBatchSpanProcessor creates a new instance of BatchSpanProcessor
// for a given export. It returns an error if exporter is nil.
// The newly created BatchSpanProcessor should then be registered with sdk
// using RegisterSpanProcessor.
func NewBatchSpanProcessor(e export.SpanBatcher, opts ...BatchSpanProcessorOption) (*BatchSpanProcessor, error) {
	if e == nil {
		return nil, errNilExporter
	}

	o := BatchSpanProcessorOptions{
		ScheduledDelayMillis: defaultScheduledDelay,
		MaxQueueSize:         defaultMaxQueueSize,
		MaxExportBatchSize:   defaultMaxExportBatchSize,
	}
	for _, opt := range opts {
		opt(&o)
	}
	bsp := &BatchSpanProcessor{
		e: e,
		o: o,
	}

	bsp.queue = make(chan *export.SpanData, bsp.o.MaxQueueSize)

	bsp.stopCh = make(chan struct{})

	// Start timer to export spans.
	ticker := time.NewTicker(bsp.o.ScheduledDelayMillis)
	bsp.stopWait.Add(1)
	go func() {
		defer ticker.Stop()
		batch := make([]*export.SpanData, 0, bsp.o.MaxExportBatchSize)
		for {
			select {
			case <-bsp.stopCh:
				bsp.processQueue(&batch)
				close(bsp.queue)
				bsp.stopWait.Done()
				return
			case <-ticker.C:
				bsp.processQueue(&batch)
			}
		}
	}()

	return bsp, nil
}

// OnStart method does nothing.
func (bsp *BatchSpanProcessor) OnStart(sd *export.SpanData) {
}

// OnEnd method enqueues export.SpanData for later processing.
func (bsp *BatchSpanProcessor) OnEnd(sd *export.SpanData) {
	bsp.enqueue(sd)
}

// Shutdown flushes the queue and waits until all spans are processed.
// It only executes once. Subsequent call does nothing.
func (bsp *BatchSpanProcessor) Shutdown() {
	bsp.stopOnce.Do(func() {
		close(bsp.stopCh)
		bsp.stopWait.Wait()
	})
}

func WithMaxQueueSize(size int) BatchSpanProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		o.MaxQueueSize = size
	}
}

func WithMaxExportBatchSize(size int) BatchSpanProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		o.MaxExportBatchSize = size
	}
}

func WithScheduleDelayMillis(delay time.Duration) BatchSpanProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		o.ScheduledDelayMillis = delay
	}
}

func WithBlocking() BatchSpanProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		o.BlockOnQueueFull = true
	}
}

// processQueue removes spans from the `queue` channel until there is
// no more data.  It calls the exporter in batches of up to
// MaxExportBatchSize until all the available data have been processed.
func (bsp *BatchSpanProcessor) processQueue(batch *[]*export.SpanData) {
	for {
		// Read spans until either the buffer fills or the
		// queue is empty.
		for ok := true; ok && len(*batch) < bsp.o.MaxExportBatchSize; {
			select {
			case sd := <-bsp.queue:
				if sd != nil && sd.SpanContext.IsSampled() {
					*batch = append(*batch, sd)
				}
			default:
				ok = false
			}
		}

		if len(*batch) == 0 {
			return
		}

		// Send one batch, then continue reading until the
		// buffer is empty.
		bsp.e.ExportSpans(context.Background(), *batch)
		*batch = (*batch)[:0]
	}
}

func (bsp *BatchSpanProcessor) enqueue(sd *export.SpanData) {
	select {
	case <-bsp.stopCh:
		return
	default:
	}
	if bsp.o.BlockOnQueueFull {
		bsp.queue <- sd
	} else {
		var ok bool
		select {
		case bsp.queue <- sd:
			ok = true
		default:
			ok = false
		}
		if !ok {
			atomic.AddUint32(&bsp.dropped, 1)
		}
	}
}
