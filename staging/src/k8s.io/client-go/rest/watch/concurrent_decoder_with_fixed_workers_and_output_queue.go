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
	"os"
	"strconv"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"
)

const (
	defaultFixedWorkersBufferedConcurrentOutputBufferSize           = 100
	defaultFixedWorkersBufferedConcurrentEnvelopeBufferSize         = 100
	defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency = 10

	concurrentDecoderWithFixedWorkersAndOutputQueueInputQueueSizeEnvVar  = "CONCURRENT_DECODER_WITH_FIXED_WORKERS_AND_OUTPUT_QUEUE_INPUT_QUEUE_SIZE"
	concurrentDecoderWithFixedWorkersAndOutputQueueOutputQueueSizeEnvVar = "CONCURRENT_DECODER_WITH_FIXED_WORKERS_AND_OUTPUT_QUEUE_OUTPUT_QUEUE_SIZE"
	concurrentDecoderWithFixedWorkersAndOutputQueueWorkerCountEnvVar     = "CONCURRENT_DECODER_WITH_FIXED_WORKERS_AND_OUTPUT_QUEUE_WORKER_COUNT"
)

type concurrentDecoderWithFixedWorkersAndOutputQueueConfig struct {
	inputQueueSize  int
	outputQueueSize int
	workerCount     int
}

func concurrentDecoderWithFixedWorkersAndOutputQueueConfigFromEnv() concurrentDecoderWithFixedWorkersAndOutputQueueConfig {
	config := concurrentDecoderWithFixedWorkersAndOutputQueueConfig{
		inputQueueSize:  defaultFixedWorkersBufferedConcurrentEnvelopeBufferSize,
		outputQueueSize: defaultFixedWorkersBufferedConcurrentOutputBufferSize,
		workerCount:     defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency,
	}
	config.inputQueueSize = intFromEnv(concurrentDecoderWithFixedWorkersAndOutputQueueInputQueueSizeEnvVar, config.inputQueueSize, 0)
	config.outputQueueSize = intFromEnv(concurrentDecoderWithFixedWorkersAndOutputQueueOutputQueueSizeEnvVar, config.outputQueueSize, 0)
	config.workerCount = intFromEnv(concurrentDecoderWithFixedWorkersAndOutputQueueWorkerCountEnvVar, config.workerCount, 1)
	return config
}

func intFromEnv(name string, defaultValue, minimum int) int {
	value, ok := os.LookupEnv(name)
	if !ok || value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < minimum {
		klog.Warningf("Ignoring environment variable %s=%q: must be an integer greater than or equal to %d", name, value, minimum)
		return defaultValue
	}
	return parsed
}

// These timings describe draining the stream, including envelope decoding and
// client backpressure. They do not isolate server or network execution time.
type fixedWorkersStreamReadStats struct {
	startedAt          time.Time
	readAt             time.Time
	envelopesRead      uint64
	streamReadDuration time.Duration
	queueFullCount     uint64
	queueWaitDuration  time.Duration
}

func (s fixedWorkersStreamReadStats) report() {
	var elapsed time.Duration
	if !s.readAt.IsZero() {
		elapsed = s.readAt.Sub(s.startedAt)
	}
	klog.InfoS("Fixed-workers watch stream read timing",
		"envelopesRead", s.envelopesRead,
		"readElapsed", elapsed,
		"streamReadAndDecodeDurationTotal", s.streamReadDuration,
		"inputQueueFullCount", s.queueFullCount,
		"inputQueueWaitDurationTotal", s.queueWaitDuration,
	)
}

type ConcurrentDecoderWithFixedWorkersAndOutputQueue struct {
	fixedWorkersBufferedConcurrentEnvelopeDecoder        *fixedWorkersBufferedConcurrentEnvelopeDecoder
	fixedWorkersBufferedConcurrentOrderedEventProcessing *fixedWorkersBufferedConcurrentOrderedEventProcessing
	wg                                                   sync.WaitGroup
	closeOnce                                            sync.Once
}

func NewConcurrentDecoderWithFixedWorkersAndOutputQueue(envelopeDecoder streaming.Decoder, embeddedDecoder runtime.Decoder) *ConcurrentDecoderWithFixedWorkersAndOutputQueue {
	config := concurrentDecoderWithFixedWorkersAndOutputQueueConfigFromEnv()
	klog.InfoS("Creating watch decoder",
		"decoder", "ConcurrentDecoderWithFixedWorkersAndOutputQueue",
		"envelopeBufferSize", config.inputQueueSize,
		"outputQueueSize", config.outputQueueSize,
		"workerCount", config.workerCount,
		"processingResultsBufferSize", config.workerCount-1,
	)
	d := &ConcurrentDecoderWithFixedWorkersAndOutputQueue{
		fixedWorkersBufferedConcurrentEnvelopeDecoder: &fixedWorkersBufferedConcurrentEnvelopeDecoder{
			envelopeDecoder:     envelopeDecoder,
			incomingEnvelopesCh: make(chan fixedWorkersBufferedConcurrentEnvelopeResult, config.inputQueueSize),
			doneCh:              make(chan struct{}),
		},
	}
	d.fixedWorkersBufferedConcurrentOrderedEventProcessing = &fixedWorkersBufferedConcurrentOrderedEventProcessing{
		incomingEnvelopesCh: d.fixedWorkersBufferedConcurrentEnvelopeDecoder.incomingEnvelopesCh,
		embeddedDecoder:     embeddedDecoder,
		outputResultsCh:     make(chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult, config.outputQueueSize),
		processingResultsCh: make(chan chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult, config.workerCount-1),
		doneCh:              make(chan struct{}),
	}
	d.wg.Add(3)
	go func() {
		defer d.wg.Done()
		d.fixedWorkersBufferedConcurrentEnvelopeDecoder.read()
	}()
	go func() {
		defer d.wg.Done()
		d.fixedWorkersBufferedConcurrentOrderedEventProcessing.scheduleEventProcessing(&d.wg, config.workerCount)
	}()
	go func() {
		defer d.wg.Done()
		d.fixedWorkersBufferedConcurrentOrderedEventProcessing.collectResults()
	}()
	return d
}

func (d *ConcurrentDecoderWithFixedWorkersAndOutputQueue) Decode() (watch.EventType, runtime.Object, error) {
	result := d.fixedWorkersBufferedConcurrentOrderedEventProcessing.nextResult()
	if result.err != nil {
		return "", nil, result.err
	}
	return result.event.Type, result.event.Object, nil
}

func (d *ConcurrentDecoderWithFixedWorkersAndOutputQueue) Close() {
	d.closeOnce.Do(func() {
		d.fixedWorkersBufferedConcurrentOrderedEventProcessing.close()
		d.fixedWorkersBufferedConcurrentEnvelopeDecoder.close()
		d.wg.Wait()
		d.fixedWorkersBufferedConcurrentEnvelopeDecoder.drain()
		d.fixedWorkersBufferedConcurrentOrderedEventProcessing.drain()
		// The reader owns the counters; waiting above makes this safe without a mutex.
		d.fixedWorkersBufferedConcurrentEnvelopeDecoder.readStats.report()
	})
}

type fixedWorkersBufferedConcurrentOrderedEventProcessingResult struct {
	event *watch.Event
	err   error
}

type fixedWorkersBufferedConcurrentOrderedEventProcessing struct {
	incomingEnvelopesCh <-chan fixedWorkersBufferedConcurrentEnvelopeResult
	embeddedDecoder     runtime.Decoder
	processingResultsCh chan chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
	outputResultsCh     chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) close() {
	p.closeOnce.Do(func() {
		close(p.doneCh)
	})
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) drain() {
	for range p.processingResultsCh {
	}
	for range p.outputResultsCh {
	}
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) nextResult() *fixedWorkersBufferedConcurrentOrderedEventProcessingResult {
	select {
	case <-p.doneCh:
		return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
	case result, ok := <-p.outputResultsCh:
		if !ok {
			return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
		}
		select {
		case <-p.doneCh:
			return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
		default:
			return result
		}
	}
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) collectResults() {
	defer close(p.outputResultsCh)
	for {
		var resultCh chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
		select {
		case <-p.doneCh:
			return
		case ch, ok := <-p.processingResultsCh:
			if !ok {
				return
			}
			resultCh = ch
		}
		var result *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
		select {
		case <-p.doneCh:
			return
		case result = <-resultCh:
		}
		select {
		case <-p.doneCh:
			return
		case p.outputResultsCh <- result:
		}
	}
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) scheduleEventProcessing(wg *sync.WaitGroup, workerCount int) {
	defer close(p.processingResultsCh)
	type job struct {
		envelopeResult fixedWorkersBufferedConcurrentEnvelopeResult
		resultCh       chan<- *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
	}
	jobs := make(chan job)
	defer close(jobs)
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			for job := range jobs {
				// Each job has a buffered result channel, so this cannot block on shutdown.
				job.resultCh <- p.processEnvelope(job.envelopeResult)
			}
		}()
	}
	for {
		var envelopeResult fixedWorkersBufferedConcurrentEnvelopeResult
		select {
		case <-p.doneCh:
			return
		case result, ok := <-p.incomingEnvelopesCh:
			if !ok {
				return
			}
			envelopeResult = result
		}
		resultCh := make(chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult, 1)
		select {
		case <-p.doneCh:
			return
		case p.processingResultsCh <- resultCh:
		}
		select {
		case <-p.doneCh:
			return
		case jobs <- job{envelopeResult: envelopeResult, resultCh: resultCh}:
		}
	}
}

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) processEnvelope(envelopeResult fixedWorkersBufferedConcurrentEnvelopeResult) *fixedWorkersBufferedConcurrentOrderedEventProcessingResult {
	if envelopeResult.err != nil {
		return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{err: envelopeResult.err}
	}
	obj, err := runtime.Decode(p.embeddedDecoder, envelopeResult.envelope.Object.Raw)
	if err != nil {
		return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{err: fmt.Errorf("unable to decode watch event: %v", err)}
	}
	return &fixedWorkersBufferedConcurrentOrderedEventProcessingResult{
		event: &watch.Event{Type: watch.EventType(envelopeResult.envelope.Type), Object: obj},
	}
}

type fixedWorkersBufferedConcurrentEnvelopeResult struct {
	envelope *metav1.WatchEvent
	err      error
}

type fixedWorkersBufferedConcurrentEnvelopeDecoder struct {
	readStats           fixedWorkersStreamReadStats
	envelopeDecoder     streaming.Decoder
	incomingEnvelopesCh chan fixedWorkersBufferedConcurrentEnvelopeResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (d *fixedWorkersBufferedConcurrentEnvelopeDecoder) close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		if err := d.envelopeDecoder.Close(); err != nil {
			utilruntime.HandleError(fmt.Errorf("close watch envelope decoder: %w", err))
		}
	})
}

func (d *fixedWorkersBufferedConcurrentEnvelopeDecoder) drain() {
	for range d.incomingEnvelopesCh {
	}
}

func (d *fixedWorkersBufferedConcurrentEnvelopeDecoder) read() {
	defer close(d.incomingEnvelopesCh)
	stats := &d.readStats
	stats.startedAt = time.Now()
	for {
		select {
		case <-d.doneCh:
			return
		default:
		}
		readStarted := time.Now()
		result := d.readEnvelope()
		// A read interrupted by Close may have waited long after the last data arrived.
		if result.err == nil {
			stats.readAt = time.Now()
			stats.streamReadDuration += stats.readAt.Sub(readStarted)
			stats.envelopesRead++
		}
		select {
		case <-d.doneCh:
			return
		case d.incomingEnvelopesCh <- result:
		default:
			// Only time sends which could not proceed immediately.
			stats.queueFullCount++
			waitStarted := time.Now()
			select {
			case <-d.doneCh:
				stats.queueWaitDuration += time.Since(waitStarted)
				return
			case d.incomingEnvelopesCh <- result:
				stats.queueWaitDuration += time.Since(waitStarted)
			}
		}
		if result.err != nil {
			return
		}
	}
}

func (d *fixedWorkersBufferedConcurrentEnvelopeDecoder) readEnvelope() fixedWorkersBufferedConcurrentEnvelopeResult {
	var got metav1.WatchEvent
	res, _, err := d.envelopeDecoder.Decode(nil, &got)
	if err != nil {
		return fixedWorkersBufferedConcurrentEnvelopeResult{err: err}
	}
	if res != &got {
		return fixedWorkersBufferedConcurrentEnvelopeResult{err: fmt.Errorf("unable to decode to metav1.WatchEvent")}
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return fixedWorkersBufferedConcurrentEnvelopeResult{err: fmt.Errorf("got invalid watch event type: %v", got.Type)}
	}

	return fixedWorkersBufferedConcurrentEnvelopeResult{envelope: &got}
}
