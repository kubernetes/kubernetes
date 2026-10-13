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
	defaultBufferedEnvelopeBufferSize           = 100
	defaultConcurrentEventProcessingConcurrency = 10
)

type ConcurrentDecoder struct {
	bufferedEnvelopeDecoder          *bufferedEnvelopeDecoder
	concurrentOrderedEventProcessing *concurrentOrderedEventProcessing
	wg                               sync.WaitGroup
}

func NewConcurrentDecoder(envelopeDecoder streaming.Decoder, embeddedDecoder runtime.Decoder) *ConcurrentDecoder {
	d := &ConcurrentDecoder{
		bufferedEnvelopeDecoder: &bufferedEnvelopeDecoder{
			envelopeDecoder:     envelopeDecoder,
			incomingEnvelopesCh: make(chan bufferedEnvelopeResult, defaultBufferedEnvelopeBufferSize),
			doneCh:              make(chan struct{}),
		},
	}
	d.concurrentOrderedEventProcessing = &concurrentOrderedEventProcessing{
		incomingEnvelopesCh: d.bufferedEnvelopeDecoder.incomingEnvelopesCh,
		embeddedDecoder:     embeddedDecoder,
		processingResultsCh: make(chan chan *concurrentOrderedEventProcessingResult, defaultConcurrentEventProcessingConcurrency-1),
		doneCh:              make(chan struct{}),
	}
	d.wg.Add(2)
	// TODO: Consider starting the reader on the first Decode call.
	go func() {
		defer d.wg.Done()
		d.bufferedEnvelopeDecoder.read()
	}()
	go func() {
		defer d.wg.Done()
		d.concurrentOrderedEventProcessing.scheduleEventProcessing(&d.wg)
	}()
	return d
}

func (d *ConcurrentDecoder) Decode() (watch.EventType, runtime.Object, error) {
	result := d.concurrentOrderedEventProcessing.nextResult()
	if result.err != nil {
		return "", nil, result.err
	}
	return result.event.Type, result.event.Object, nil
}

func (d *ConcurrentDecoder) Close() {
	d.concurrentOrderedEventProcessing.close()
	d.bufferedEnvelopeDecoder.close()
	d.wg.Wait()
	d.bufferedEnvelopeDecoder.drain()
	d.concurrentOrderedEventProcessing.drain()
}

type concurrentOrderedEventProcessingResult struct {
	event *watch.Event
	err   error
}

type concurrentOrderedEventProcessing struct {
	incomingEnvelopesCh <-chan bufferedEnvelopeResult
	embeddedDecoder     runtime.Decoder
	processingResultsCh chan chan *concurrentOrderedEventProcessingResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (p *concurrentOrderedEventProcessing) close() {
	p.closeOnce.Do(func() {
		close(p.doneCh)
	})
}

func (p *concurrentOrderedEventProcessing) drain() {
	for range p.processingResultsCh {
	}
}

func (p *concurrentOrderedEventProcessing) nextResult() *concurrentOrderedEventProcessingResult {
	select {
	case <-p.doneCh:
		return &concurrentOrderedEventProcessingResult{err: io.EOF}
	case resultCh, ok := <-p.processingResultsCh:
		if !ok {
			return &concurrentOrderedEventProcessingResult{err: io.EOF}
		}
		select {
		case <-p.doneCh:
			return &concurrentOrderedEventProcessingResult{err: io.EOF}
		case result := <-resultCh:
			select {
			case <-p.doneCh:
				return &concurrentOrderedEventProcessingResult{err: io.EOF}
			default:
				return result
			}
		}
	}
}

func (p *concurrentOrderedEventProcessing) scheduleEventProcessing(wg *sync.WaitGroup) {
	defer close(p.processingResultsCh)
	for {
		var envelopeResult bufferedEnvelopeResult
		select {
		case <-p.doneCh:
			return
		case result, ok := <-p.incomingEnvelopesCh:
			if !ok {
				return
			}
			envelopeResult = result
		}
		resultCh := make(chan *concurrentOrderedEventProcessingResult, 1)
		select {
		case <-p.doneCh:
			return
		case p.processingResultsCh <- resultCh:
		}
		// The scheduler stays registered while it can add workers, so Close cannot finish early.
		wg.Add(1)
		go func(envelopeResult bufferedEnvelopeResult, resultCh chan<- *concurrentOrderedEventProcessingResult) {
			defer wg.Done()
			result := p.processEnvelope(envelopeResult)
			select {
			case <-p.doneCh:
			case resultCh <- result:
			}
		}(envelopeResult, resultCh)
	}
}

func (p *concurrentOrderedEventProcessing) processEnvelope(envelopeResult bufferedEnvelopeResult) *concurrentOrderedEventProcessingResult {
	if envelopeResult.err != nil {
		return &concurrentOrderedEventProcessingResult{err: envelopeResult.err}
	}
	obj, err := runtime.Decode(p.embeddedDecoder, envelopeResult.envelope.Object.Raw)
	if err != nil {
		return &concurrentOrderedEventProcessingResult{err: fmt.Errorf("unable to decode watch event: %v", err)}
	}
	return &concurrentOrderedEventProcessingResult{
		event: &watch.Event{Type: watch.EventType(envelopeResult.envelope.Type), Object: obj},
	}
}

type bufferedEnvelopeResult struct {
	envelope *metav1.WatchEvent
	err      error
}

type bufferedEnvelopeDecoder struct {
	envelopeDecoder     streaming.Decoder
	incomingEnvelopesCh chan bufferedEnvelopeResult
	doneCh              chan struct{}
	closeOnce           sync.Once
}

func (d *bufferedEnvelopeDecoder) close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		if err := d.envelopeDecoder.Close(); err != nil {
			utilruntime.HandleError(fmt.Errorf("close watch envelope decoder: %w", err))
		}
	})
}

func (d *bufferedEnvelopeDecoder) drain() {
	for range d.incomingEnvelopesCh {
	}
}

func (d *bufferedEnvelopeDecoder) read() {
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

func (d *bufferedEnvelopeDecoder) readEnvelope() bufferedEnvelopeResult {
	var got metav1.WatchEvent
	res, _, err := d.envelopeDecoder.Decode(nil, &got)
	if err != nil {
		return bufferedEnvelopeResult{err: err}
	}
	if res != &got {
		return bufferedEnvelopeResult{err: fmt.Errorf("unable to decode to metav1.WatchEvent")}
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return bufferedEnvelopeResult{err: fmt.Errorf("got invalid watch event type: %v", got.Type)}
	}

	return bufferedEnvelopeResult{envelope: &got}
}
