/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package watch

import (
	"fmt"
	"io"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

type rawWatchEventResult struct {
	event *metav1.WatchEvent
	err   error
}

type decodedWatchEventResult struct {
	event *watch.Event
	err   error
}

const (
	defaultDecodedResultQueueSize = 100
	defaultRawResultQueueSize     = 100
	defaultDecodingConcurrency    = 10
)

// ConcurrentWatchEventDecoder decodes watch events concurrently while
// returning them in the order they were read.
type ConcurrentWatchEventDecoder struct {
	eventDecoder         *bufferedWatchEventDecoder
	objectDecoder        runtime.Decoder
	pendingResultsCh     chan chan *decodedWatchEventResult
	decodedResultsCh     chan *decodedWatchEventResult
	doneCh               chan struct{}
	closeOnce            sync.Once
	waitGroup            sync.WaitGroup
	decodeLogger         *watch.BlockLogger
	readerBlockLogger    *watch.BlockLogger
	schedulerBlockLogger *watch.BlockLogger
	collectorBlockLogger *watch.BlockLogger
}

func NewConcurrentWatchEventDecoder(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder) *ConcurrentWatchEventDecoder {
	return NewConcurrentWatchEventDecoderWithLogger(streamDecoder, objectDecoder, nil)
}

// NewConcurrentWatchEventDecoderWithLogger creates a ConcurrentWatchEventDecoder carrying logger's contextual fields.
func NewConcurrentWatchEventDecoderWithLogger(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder, logger *klog.Logger) *ConcurrentWatchEventDecoder {
	return newConcurrentWatchEventDecoderWithLogger(streamDecoder, objectDecoder, defaultRawResultQueueSize, defaultDecodedResultQueueSize, defaultDecodingConcurrency, logger, "", clock.RealClock{})
}

func newConcurrentWatchEventDecoder(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder, rawResultQueueSize, decodedResultQueueSize, decodingConcurrency int) *ConcurrentWatchEventDecoder {
	return newConcurrentWatchEventDecoderWithLogger(streamDecoder, objectDecoder, rawResultQueueSize, decodedResultQueueSize, decodingConcurrency, nil, "", clock.RealClock{})
}

func newConcurrentStageLogger(baseLogger *klog.Logger, source, stage, msg string, threshold time.Duration, clk clock.Clock) *watch.BlockLogger {
	l := klog.Background()
	if baseLogger != nil {
		l = *baseLogger
	}
	stageLogger := l.WithValues("stage", stage)
	return watch.NewBlockLoggerWithOptions(msg, watch.BlockLoggerOptions{
		Threshold:       threshold,
		SingleThreshold: watch.DefaultBlockLoggerSingleThreshold,
		Source:          source,
		Logger:          &stageLogger,
		Clock:           clk,
	})
}

func newConcurrentWatchEventDecoderWithLogger(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder, rawResultQueueSize, decodedResultQueueSize, decodingConcurrency int, logger *klog.Logger, source string, clk clock.Clock) *ConcurrentWatchEventDecoder {
	if clk == nil {
		clk = clock.RealClock{}
	}
	readerBlockLogger := newConcurrentStageLogger(
		logger, source, "reader",
		"Fast watch stream, slow concurrent decode reader handoff. rawResultsCh is full and stream reads are stalled",
		watch.DefaultBlockLoggerThreshold, clk,
	)
	d := &ConcurrentWatchEventDecoder{
		eventDecoder: &bufferedWatchEventDecoder{
			streamDecoder:     streamDecoder,
			rawResultsCh:      make(chan rawWatchEventResult, rawResultQueueSize),
			doneCh:            make(chan struct{}),
			readerBlockLogger: readerBlockLogger,
		},
		objectDecoder:    objectDecoder,
		decodedResultsCh: make(chan *decodedWatchEventResult, decodedResultQueueSize),
		pendingResultsCh: make(chan chan *decodedWatchEventResult, decodingConcurrency-1),
		doneCh:           make(chan struct{}),
		decodeLogger: newConcurrentStageLogger(
			logger, source, "collector-wait",
			"Fast watcher, slow object decoding. Probably caused by CPU-intensive unmarshaling or large watch events",
			watch.DefaultDecodeLoggerThreshold, clk,
		),
		readerBlockLogger: readerBlockLogger,
		schedulerBlockLogger: newConcurrentStageLogger(
			logger, source, "scheduler",
			"Fast watch stream, slow concurrent decode scheduler. Decode concurrency slots are saturated",
			watch.DefaultBlockLoggerThreshold, clk,
		),
		collectorBlockLogger: newConcurrentStageLogger(
			logger, source, "collector-handoff",
			"Fast watch stream, slow concurrent decode collector handoff. decodedResultsCh is full and consumer is stalled",
			watch.DefaultBlockLoggerThreshold, clk,
		),
	}
	d.waitGroup.Add(3)
	go func() {
		defer utilruntime.HandleCrash()
		defer d.waitGroup.Done()
		d.eventDecoder.read()
	}()
	go func() {
		defer utilruntime.HandleCrash()
		defer d.waitGroup.Done()
		d.scheduleDecoding()
	}()
	go func() {
		defer utilruntime.HandleCrash()
		defer d.waitGroup.Done()
		d.collectResults()
	}()
	return d
}

func (d *ConcurrentWatchEventDecoder) Decode() (watch.EventType, runtime.Object, error) {
	select {
	case <-d.doneCh:
		return "", nil, io.EOF
	case result, ok := <-d.decodedResultsCh:
		if !ok {
			return "", nil, io.EOF
		}
		if result.err != nil {
			return "", nil, result.err
		}
		return result.event.Type, result.event.Object, nil
	}
}

func (d *ConcurrentWatchEventDecoder) Close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		d.eventDecoder.close()
	})
	d.waitGroup.Wait()
}

func (d *ConcurrentWatchEventDecoder) collectResults() {
	defer close(d.decodedResultsCh)
	defer d.decodeLogger.Flush()
	defer d.collectorBlockLogger.Flush()
	var resultCh chan *decodedWatchEventResult
	var ok bool

	for {
		select {
		case <-d.doneCh:
			return
		case resultCh, ok = <-d.pendingResultsCh:
			if !ok {
				return
			}
		}

		// Decodes overlap, so summing their durations can exceed wall time. Time how long the
		// ordered output waits on the head-of-line decode instead: the wait is serial, and
		// without overlap it reduces to the decode time the non-concurrent Decoder records.
		start := d.decodeLogger.Now()
		result := <-resultCh
		// Skip errored results: they also carry stream read failures, which never reach
		// object decoding.
		if result.err == nil {
			if result.event.Type != watch.Error {
				d.decodeLogger.EnsureResourceType(result.event.Object)
				d.readerBlockLogger.EnsureResourceType(result.event.Object)
				d.schedulerBlockLogger.EnsureResourceType(result.event.Object)
				d.collectorBlockLogger.EnsureResourceType(result.event.Object)
			}
			d.decodeLogger.RecordSince(start)
		}

		handoffStart := d.collectorBlockLogger.Now()
		select {
		case <-d.doneCh:
			d.collectorBlockLogger.RecordSince(handoffStart)
			return
		case d.decodedResultsCh <- result:
			d.collectorBlockLogger.RecordSince(handoffStart)
		}
	}
}

func (d *ConcurrentWatchEventDecoder) scheduleDecoding() {
	defer close(d.pendingResultsCh)
	defer d.schedulerBlockLogger.Flush()
	var eventResult rawWatchEventResult
	var ok bool

	for {
		select {
		case <-d.doneCh:
			return
		case eventResult, ok = <-d.eventDecoder.rawResultsCh:
			if !ok {
				return
			}
		}
		resultCh := make(chan *decodedWatchEventResult, 1)
		schedStart := d.schedulerBlockLogger.Now()
		select {
		case <-d.doneCh:
			d.schedulerBlockLogger.RecordSince(schedStart)
			return
		case d.pendingResultsCh <- resultCh:
			d.schedulerBlockLogger.RecordSince(schedStart)
		}
		d.waitGroup.Add(1)
		go func(eventResult rawWatchEventResult) {
			defer d.waitGroup.Done()
			defer utilruntime.HandleCrash(func(r interface{}) {
				resultCh <- &decodedWatchEventResult{
					err: fmt.Errorf("panic decoding watch event object: %v", r),
				}
			})
			resultCh <- d.decodeEvent(eventResult)
		}(eventResult)
	}
}

func (d *ConcurrentWatchEventDecoder) decodeEvent(eventResult rawWatchEventResult) *decodedWatchEventResult {
	if eventResult.err != nil {
		return &decodedWatchEventResult{err: eventResult.err}
	}
	obj, err := runtime.Decode(d.objectDecoder, eventResult.event.Object.Raw)
	if err != nil {
		return &decodedWatchEventResult{err: fmt.Errorf("unable to decode watch event: %w", err)}
	}
	return &decodedWatchEventResult{
		event: &watch.Event{Type: watch.EventType(eventResult.event.Type), Object: obj},
	}
}

type bufferedWatchEventDecoder struct {
	streamDecoder     streaming.Decoder
	rawResultsCh      chan rawWatchEventResult
	doneCh            chan struct{}
	readerBlockLogger *watch.BlockLogger
}

func (d *bufferedWatchEventDecoder) close() {
	close(d.doneCh)
	if err := d.streamDecoder.Close(); err != nil {
		utilruntime.HandleError(fmt.Errorf("close watch event decoder: %w", err))
	}
}

func (d *bufferedWatchEventDecoder) read() {
	defer close(d.rawResultsCh)
	defer d.readerBlockLogger.Flush()
	for {
		select {
		case <-d.doneCh:
			return
		default:
		}

		got, err := readEvent(d.streamDecoder)
		result := rawWatchEventResult{event: &got, err: err}
		enqueueStart := d.readerBlockLogger.Now()
		select {
		case <-d.doneCh:
			d.readerBlockLogger.RecordSince(enqueueStart)
			return
		case d.rawResultsCh <- result:
			d.readerBlockLogger.RecordSince(enqueueStart)
		}
	}
}
