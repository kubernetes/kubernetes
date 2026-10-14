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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

const (
	defaultFixedWorkersBufferedConcurrentOutputBufferSize           = 100
	defaultFixedWorkersBufferedConcurrentEnvelopeBufferSize         = 100
	defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency = 10
)

type ConcurrentDecoderWithFixedWorkersAndOutputQueue struct {
	fixedWorkersBufferedConcurrentEnvelopeDecoder        *fixedWorkersBufferedConcurrentEnvelopeDecoder
	fixedWorkersBufferedConcurrentOrderedEventProcessing *fixedWorkersBufferedConcurrentOrderedEventProcessing
	wg                                                   sync.WaitGroup
}

func NewConcurrentDecoderWithFixedWorkersAndOutputQueue(envelopeDecoder streaming.Decoder, embeddedDecoder runtime.Decoder) *ConcurrentDecoderWithFixedWorkersAndOutputQueue {
	d := &ConcurrentDecoderWithFixedWorkersAndOutputQueue{
		fixedWorkersBufferedConcurrentEnvelopeDecoder: &fixedWorkersBufferedConcurrentEnvelopeDecoder{
			envelopeDecoder:     envelopeDecoder,
			incomingEnvelopesCh: make(chan fixedWorkersBufferedConcurrentEnvelopeResult, defaultFixedWorkersBufferedConcurrentEnvelopeBufferSize),
			doneCh:              make(chan struct{}),
		},
	}
	d.fixedWorkersBufferedConcurrentOrderedEventProcessing = &fixedWorkersBufferedConcurrentOrderedEventProcessing{
		incomingEnvelopesCh: d.fixedWorkersBufferedConcurrentEnvelopeDecoder.incomingEnvelopesCh,
		embeddedDecoder:     embeddedDecoder,
		outputResultsCh:     make(chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult, defaultFixedWorkersBufferedConcurrentOutputBufferSize),
		processingResultsCh: make(chan chan *fixedWorkersBufferedConcurrentOrderedEventProcessingResult, defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency-1),
		doneCh:              make(chan struct{}),
	}
	d.wg.Add(3)
	go func() {
		defer d.wg.Done()
		d.fixedWorkersBufferedConcurrentEnvelopeDecoder.read()
	}()
	go func() {
		defer d.wg.Done()
		d.fixedWorkersBufferedConcurrentOrderedEventProcessing.scheduleEventProcessing(&d.wg)
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
	d.fixedWorkersBufferedConcurrentOrderedEventProcessing.close()
	d.fixedWorkersBufferedConcurrentEnvelopeDecoder.close()
	d.wg.Wait()
	d.fixedWorkersBufferedConcurrentEnvelopeDecoder.drain()
	d.fixedWorkersBufferedConcurrentOrderedEventProcessing.drain()
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

func (p *fixedWorkersBufferedConcurrentOrderedEventProcessing) scheduleEventProcessing(wg *sync.WaitGroup) {
	defer close(p.processingResultsCh)
	type job struct {
		envelopeResult fixedWorkersBufferedConcurrentEnvelopeResult
		resultCh       chan<- *fixedWorkersBufferedConcurrentOrderedEventProcessingResult
	}
	jobs := make(chan job)
	defer close(jobs)
	wg.Add(defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency)
	for i := 0; i < defaultFixedWorkersBufferedConcurrentEventProcessingConcurrency; i++ {
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
	for {
		select {
		case <-d.doneCh:
			return
		default:
		}

		result := d.readEnvelope()
		select {
		case <-d.doneCh:
			return
		case d.incomingEnvelopesCh <- result:
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
