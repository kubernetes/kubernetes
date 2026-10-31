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
	eventDecoder     *bufferedWatchEventDecoder
	objectDecoder    runtime.Decoder
	pendingResultsCh chan chan *decodedWatchEventResult
	decodedResultsCh chan *decodedWatchEventResult
	doneCh           chan struct{}
	closeOnce        sync.Once
	waitGroup        sync.WaitGroup
}

func NewConcurrentWatchEventDecoder(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder) *ConcurrentWatchEventDecoder {
	return newConcurrentWatchEventDecoder(streamDecoder, objectDecoder, defaultRawResultQueueSize, defaultDecodedResultQueueSize, defaultDecodingConcurrency)
}

func newConcurrentWatchEventDecoder(streamDecoder streaming.Decoder, objectDecoder runtime.Decoder, rawResultQueueSize, decodedResultQueueSize, decodingConcurrency int) *ConcurrentWatchEventDecoder {
	d := &ConcurrentWatchEventDecoder{
		eventDecoder: &bufferedWatchEventDecoder{
			streamDecoder: streamDecoder,
			rawResultsCh:  make(chan rawWatchEventResult, rawResultQueueSize),
			doneCh:        make(chan struct{}),
		},
		objectDecoder:    objectDecoder,
		decodedResultsCh: make(chan *decodedWatchEventResult, decodedResultQueueSize),
		pendingResultsCh: make(chan chan *decodedWatchEventResult, decodingConcurrency-1),
		doneCh:           make(chan struct{}),
	}
	d.waitGroup.Add(3)
	go func() {
		defer d.waitGroup.Done()
		d.eventDecoder.read()
	}()
	go func() {
		defer d.waitGroup.Done()
		d.scheduleDecoding()
	}()
	go func() {
		defer d.waitGroup.Done()
		d.collectResults()
	}()
	return d
}

func (d *ConcurrentWatchEventDecoder) Decode() (watch.EventType, runtime.Object, error) {
	result := d.nextResult()
	if result.err != nil {
		return "", nil, result.err
	}
	return result.event.Type, result.event.Object, nil
}

func (d *ConcurrentWatchEventDecoder) Close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		d.eventDecoder.close()
	})
	d.waitGroup.Wait()
}

func (d *ConcurrentWatchEventDecoder) nextResult() *decodedWatchEventResult {
	select {
	case <-d.doneCh:
		return &decodedWatchEventResult{err: io.EOF}
	case result, ok := <-d.decodedResultsCh:
		if !ok {
			return &decodedWatchEventResult{err: io.EOF}
		}
		select {
		case <-d.doneCh:
			return &decodedWatchEventResult{err: io.EOF}
		default:
			return result
		}
	}
}

func (d *ConcurrentWatchEventDecoder) collectResults() {
	defer close(d.decodedResultsCh)
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
		var result *decodedWatchEventResult
		select {
		case <-d.doneCh:
			return
		case result = <-resultCh:
		}
		select {
		case <-d.doneCh:
			return
		case d.decodedResultsCh <- result:
		}
	}
}

func (d *ConcurrentWatchEventDecoder) scheduleDecoding() {
	defer close(d.pendingResultsCh)
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
		select {
		case <-d.doneCh:
			return
		case d.pendingResultsCh <- resultCh:
		}
		d.waitGroup.Add(1)
		go func(eventResult rawWatchEventResult) {
			defer d.waitGroup.Done()
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
	streamDecoder streaming.Decoder
	rawResultsCh  chan rawWatchEventResult
	doneCh        chan struct{}
}

func (d *bufferedWatchEventDecoder) close() {
	close(d.doneCh)
	if err := d.streamDecoder.Close(); err != nil {
		utilruntime.HandleError(fmt.Errorf("close watch event decoder: %w", err))
	}
}

func (d *bufferedWatchEventDecoder) read() {
	defer close(d.rawResultsCh)
	for {
		select {
		case <-d.doneCh:
			return
		default:
		}

		result := d.readEvent()
		select {
		case <-d.doneCh:
			return
		case d.rawResultsCh <- result:
		}
		// Watch events are ordered, so there is no value in processing anything
		// that follows an error.
		if result.err != nil {
			return
		}
	}
}

func (d *bufferedWatchEventDecoder) readEvent() rawWatchEventResult {
	got, err := readEvent(d.streamDecoder)
	if err != nil {
		return rawWatchEventResult{err: err}
	}
	return rawWatchEventResult{event: &got}
}
