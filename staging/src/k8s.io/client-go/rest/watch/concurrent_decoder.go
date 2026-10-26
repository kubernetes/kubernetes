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

type bufferedConcurrentWatchEventProcessor struct {
	rawResultsCh     <-chan rawWatchEventResult
	objectDecoder    runtime.Decoder
	pendingResultsCh chan chan *decodedWatchEventResult
	decodedResultsCh chan *decodedWatchEventResult
	doneCh           chan struct{}
	closeOnce        sync.Once
}

func (p *bufferedConcurrentWatchEventProcessor) close() {
	p.closeOnce.Do(func() {
		close(p.doneCh)
	})
}

func (p *bufferedConcurrentWatchEventProcessor) nextResult() *decodedWatchEventResult {
	select {
	case <-p.doneCh:
		return &decodedWatchEventResult{err: io.EOF}
	case result, ok := <-p.decodedResultsCh:
		if !ok {
			return &decodedWatchEventResult{err: io.EOF}
		}
		select {
		case <-p.doneCh:
			return &decodedWatchEventResult{err: io.EOF}
		default:
			return result
		}
	}
}

func (p *bufferedConcurrentWatchEventProcessor) collectResults() {
	defer close(p.decodedResultsCh)
	var resultCh chan *decodedWatchEventResult
	var ok bool

	for {
		select {
		case <-p.doneCh:
			return
		case resultCh, ok = <-p.pendingResultsCh:
			if !ok {
				return
			}
		}
		var result *decodedWatchEventResult
		select {
		case <-p.doneCh:
			return
		case result = <-resultCh:
		}
		select {
		case <-p.doneCh:
			return
		case p.decodedResultsCh <- result:
		}
	}
}

func (p *bufferedConcurrentWatchEventProcessor) scheduleDecoding(waitGroup *sync.WaitGroup) {
	defer close(p.pendingResultsCh)
	for {
		var eventResult rawWatchEventResult
		select {
		case <-p.doneCh:
			return
		case result, ok := <-p.rawResultsCh:
			if !ok {
				return
			}
			eventResult = result
		}
		resultCh := make(chan *decodedWatchEventResult, 1)
		select {
		case <-p.doneCh:
			return
		case p.pendingResultsCh <- resultCh:
		}
		waitGroup.Add(1)
		go func(eventResult rawWatchEventResult) {
			defer waitGroup.Done()
			resultCh <- p.decodeEvent(eventResult)
		}(eventResult)
	}
}

func (p *bufferedConcurrentWatchEventProcessor) decodeEvent(eventResult rawWatchEventResult) *decodedWatchEventResult {
	if eventResult.err != nil {
		return &decodedWatchEventResult{err: eventResult.err}
	}
	obj, err := runtime.Decode(p.objectDecoder, eventResult.event.Object.Raw)
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
	closeOnce     sync.Once
}

func (d *bufferedWatchEventDecoder) close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		if err := d.streamDecoder.Close(); err != nil {
			utilruntime.HandleError(fmt.Errorf("close watch event decoder: %w", err))
		}
	})
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
