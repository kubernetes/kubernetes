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
	"k8s.io/klog/v2"
)

const (
	defaultBufferedConcurrentOutputBufferSize           = 100
	defaultBufferedConcurrentEnvelopeBufferSize         = 100
	defaultBufferedConcurrentEventProcessingConcurrency = 10
)

type ConcurrentDecoderWithOutputQueue struct {
	bufferedConcurrentEnvelopeDecoder        *bufferedConcurrentEnvelopeDecoder
	bufferedConcurrentOrderedEventProcessing *bufferedConcurrentOrderedEventProcessing
	wg                                       sync.WaitGroup
}

func NewConcurrentDecoderWithOutputQueue(envelopeDecoder streaming.Decoder, embeddedDecoder runtime.Decoder) *ConcurrentDecoderWithOutputQueue {
	klog.InfoS("Creating watch decoder",
		"decoder", "ConcurrentDecoderWithOutputQueue",
		"envelopeBufferSize", defaultBufferedConcurrentEnvelopeBufferSize,
		"outputQueueSize", defaultBufferedConcurrentOutputBufferSize,
		"processingResultsBufferSize", defaultBufferedConcurrentEventProcessingConcurrency-1,
	)
	d := &ConcurrentDecoderWithOutputQueue{
		bufferedConcurrentEnvelopeDecoder: &bufferedConcurrentEnvelopeDecoder{
			envelopeDecoder:     envelopeDecoder,
			incomingEnvelopesCh: make(chan bufferedConcurrentEnvelopeResult, defaultBufferedConcurrentEnvelopeBufferSize),
			doneCh:              make(chan struct{}),
		},
	}
	d.bufferedConcurrentOrderedEventProcessing = &bufferedConcurrentOrderedEventProcessing{
		incomingEnvelopesCh: d.bufferedConcurrentEnvelopeDecoder.incomingEnvelopesCh,
		embeddedDecoder:     embeddedDecoder,
		outputResultsCh:     make(chan *bufferedConcurrentOrderedEventProcessingResult, defaultBufferedConcurrentOutputBufferSize),
		processingResultsCh: make(chan chan *bufferedConcurrentOrderedEventProcessingResult, defaultBufferedConcurrentEventProcessingConcurrency-1),
		doneCh:              make(chan struct{}),
	}
	d.wg.Add(3)
	go func() {
		defer d.wg.Done()
		d.bufferedConcurrentEnvelopeDecoder.read()
	}()
	go func() {
		defer d.wg.Done()
		d.bufferedConcurrentOrderedEventProcessing.scheduleEventProcessing(&d.wg)
	}()
	go func() {
		defer d.wg.Done()
		d.bufferedConcurrentOrderedEventProcessing.collectResults()
	}()
	return d
}

func (d *ConcurrentDecoderWithOutputQueue) Decode() (watch.EventType, runtime.Object, error) {
	result := d.bufferedConcurrentOrderedEventProcessing.nextResult()
	if result.err != nil {
		return "", nil, result.err
	}
	return result.event.Type, result.event.Object, nil
}

func (d *ConcurrentDecoderWithOutputQueue) Close() {
	d.bufferedConcurrentOrderedEventProcessing.close()
	d.bufferedConcurrentEnvelopeDecoder.close()
	d.wg.Wait()
	d.bufferedConcurrentEnvelopeDecoder.drain()
	d.bufferedConcurrentOrderedEventProcessing.drain()
}

type bufferedConcurrentOrderedEventProcessingResult struct {
	event *watch.Event
	err   error
}

type bufferedConcurrentOrderedEventProcessing struct {
	incomingEnvelopesCh <-chan bufferedConcurrentEnvelopeResult
	embeddedDecoder     runtime.Decoder
	processingResultsCh chan chan *bufferedConcurrentOrderedEventProcessingResult
	outputResultsCh     chan *bufferedConcurrentOrderedEventProcessingResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (p *bufferedConcurrentOrderedEventProcessing) close() {
	p.closeOnce.Do(func() {
		close(p.doneCh)
	})
}

func (p *bufferedConcurrentOrderedEventProcessing) drain() {
	for range p.processingResultsCh {
	}
	for range p.outputResultsCh {
	}
}

func (p *bufferedConcurrentOrderedEventProcessing) nextResult() *bufferedConcurrentOrderedEventProcessingResult {
	select {
	case <-p.doneCh:
		return &bufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
	case result, ok := <-p.outputResultsCh:
		if !ok {
			return &bufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
		}
		select {
		case <-p.doneCh:
			return &bufferedConcurrentOrderedEventProcessingResult{err: io.EOF}
		default:
			return result
		}
	}
}

func (p *bufferedConcurrentOrderedEventProcessing) collectResults() {
	defer close(p.outputResultsCh)
	for {
		var resultCh chan *bufferedConcurrentOrderedEventProcessingResult
		select {
		case <-p.doneCh:
			return
		case ch, ok := <-p.processingResultsCh:
			if !ok {
				return
			}
			resultCh = ch
		}
		var result *bufferedConcurrentOrderedEventProcessingResult
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

func (p *bufferedConcurrentOrderedEventProcessing) scheduleEventProcessing(wg *sync.WaitGroup) {
	defer close(p.processingResultsCh)
	for {
		var envelopeResult bufferedConcurrentEnvelopeResult
		select {
		case <-p.doneCh:
			return
		case result, ok := <-p.incomingEnvelopesCh:
			if !ok {
				return
			}
			envelopeResult = result
		}
		resultCh := make(chan *bufferedConcurrentOrderedEventProcessingResult, 1)
		select {
		case <-p.doneCh:
			return
		case p.processingResultsCh <- resultCh:
		}
		wg.Add(1)
		go func(envelopeResult bufferedConcurrentEnvelopeResult, resultCh chan<- *bufferedConcurrentOrderedEventProcessingResult) {
			defer wg.Done()
			result := p.processEnvelope(envelopeResult)
			select {
			case <-p.doneCh:
			case resultCh <- result:
			}
		}(envelopeResult, resultCh)
	}
}

func (p *bufferedConcurrentOrderedEventProcessing) processEnvelope(envelopeResult bufferedConcurrentEnvelopeResult) *bufferedConcurrentOrderedEventProcessingResult {
	if envelopeResult.err != nil {
		return &bufferedConcurrentOrderedEventProcessingResult{err: envelopeResult.err}
	}
	obj, err := runtime.Decode(p.embeddedDecoder, envelopeResult.envelope.Object.Raw)
	if err != nil {
		return &bufferedConcurrentOrderedEventProcessingResult{err: fmt.Errorf("unable to decode watch event: %v", err)}
	}
	return &bufferedConcurrentOrderedEventProcessingResult{
		event: &watch.Event{Type: watch.EventType(envelopeResult.envelope.Type), Object: obj},
	}
}

type bufferedConcurrentEnvelopeResult struct {
	envelope *metav1.WatchEvent
	err      error
}

type bufferedConcurrentEnvelopeDecoder struct {
	envelopeDecoder     streaming.Decoder
	incomingEnvelopesCh chan bufferedConcurrentEnvelopeResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (d *bufferedConcurrentEnvelopeDecoder) close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		if err := d.envelopeDecoder.Close(); err != nil {
			utilruntime.HandleError(fmt.Errorf("close watch envelope decoder: %w", err))
		}
	})
}

func (d *bufferedConcurrentEnvelopeDecoder) drain() {
	for range d.incomingEnvelopesCh {
	}
}

func (d *bufferedConcurrentEnvelopeDecoder) read() {
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

func (d *bufferedConcurrentEnvelopeDecoder) readEnvelope() bufferedConcurrentEnvelopeResult {
	var got metav1.WatchEvent
	res, _, err := d.envelopeDecoder.Decode(nil, &got)
	if err != nil {
		return bufferedConcurrentEnvelopeResult{err: err}
	}
	if res != &got {
		return bufferedConcurrentEnvelopeResult{err: fmt.Errorf("unable to decode to metav1.WatchEvent")}
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return bufferedConcurrentEnvelopeResult{err: fmt.Errorf("got invalid watch event type: %v", got.Type)}
	}

	return bufferedConcurrentEnvelopeResult{envelope: &got}
}
