/*
Copyright 2014 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
)

const processEventConcurrency = 10

type eventData struct {
	action string
	object runtime.Object
	err    error
}

// Decoder implements the watch.Decoder interface for io.ReadClosers that
// have contents which consist of a series of watchEvent objects encoded
// with the given streaming decoder. The internal objects will be then
// decoded by the embedded decoder.
type Decoder struct {
	decoder         streaming.Decoder
	embeddedDecoder runtime.Decoder
	getEvent        func() (watch.EventType, runtime.Object, error)
	stopCh          chan struct{}
	processing      concurrentOrderedEventProcessing
}

// NewDecoder creates an Decoder for the given writer and codec.
func NewDecoder(decoder streaming.Decoder, embeddedDecoder runtime.Decoder) *Decoder {
	d := &Decoder{
		decoder:         decoder,
		embeddedDecoder: embeddedDecoder,
		stopCh:          make(chan struct{}),
	}
	d.getEvent = func() (watch.EventType, runtime.Object, error) {
		var got metav1.WatchEvent
		got, err := decodeWatchEvent(d.decoder)
		if err != nil {
			return "", nil, err
		}

		obj, err := runtime.Decode(d.embeddedDecoder, got.Object.Raw)
		if err != nil {
			return "", nil, fmt.Errorf("unable to decode watch event: %v", err)
		}
		return watch.EventType(got.Type), obj, nil
	}
	if clientfeatures.FeatureGates().Enabled(clientfeatures.ClientsConcurrentWatchObjectDecode) {
		d.startConcurrentDecode()
		d.getEvent = func() (watch.EventType, runtime.Object, error) {
			select {
			case <-d.stopCh:
				return "", nil, io.EOF
			case event, ok := <-d.processing.resultChan:
				if !ok {
					// should not happen. eventChan is closed after error, when got an error, getEvent loop will exit.
					return "", nil, io.EOF
				}
				return watch.EventType(event.action), event.object, event.err
			}
		}
	}
	return d
}

// Decode blocks until it can return the next object in the reader. Returns an error
// if the reader is closed or an object can't be decoded.
func (d *Decoder) Decode() (watch.EventType, runtime.Object, error) {
	return d.getEvent()
}

func decodeWatchEvent(decoder streaming.Decoder) (metav1.WatchEvent, error) {
	var got metav1.WatchEvent
	res, _, err := decoder.Decode(nil, &got)
	if err != nil {
		return metav1.WatchEvent{}, err
	}
	if res != &got {
		return metav1.WatchEvent{}, fmt.Errorf("unable to decode to metav1.WatchEvent")
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return metav1.WatchEvent{}, fmt.Errorf("got invalid watch event type: %v", got.Type)
	}
	return got, nil
}

func (d *Decoder) startConcurrentDecode() {
	d.processing = concurrentOrderedEventProcessing{
		resultChan:      make(chan *eventData, processEventConcurrency-1),
		processingQueue: make(chan chan *eventData, processEventConcurrency-1),
		decoder:         d.decoder,
		embeddedDecoder: d.embeddedDecoder,
	}
	go func() {
		d.processing.scheduleEventProcessing(d.stopCh)
	}()
	go func() {
		d.processing.collectEventProcessing(d.stopCh)
	}()

}

type concurrentOrderedEventProcessing struct {
	resultChan      chan *eventData
	processingQueue chan chan *eventData
	decoder         streaming.Decoder
	embeddedDecoder runtime.Decoder
}

func (p *concurrentOrderedEventProcessing) scheduleEventProcessing(stopCh <-chan struct{}) {
	for {
		got, err := decodeWatchEvent(p.decoder)
		processingResponse := make(chan *eventData, 1)
		select {
		case <-stopCh:
			return
		case p.processingQueue <- processingResponse:
		}

		go func(watchEvent *metav1.WatchEvent, err error, response chan<- *eventData) {
			pr := &eventData{
				action: watchEvent.Type,
			}
			if err != nil {
				pr.err = err
			} else {
				obj, err := runtime.Decode(p.embeddedDecoder, got.Object.Raw)
				if err != nil {
					pr.err = fmt.Errorf("unable to decode watch event: %v", err)
				} else {
					pr.object = obj
				}
			}
			response <- pr
		}(&got, err, processingResponse)
		if err != nil {
			return
		}
	}
}

func (p *concurrentOrderedEventProcessing) collectEventProcessing(stopCh <-chan struct{}) {
	defer close(p.resultChan)
	for {
		var processingResponse chan *eventData
		var ok bool
		select {
		case <-stopCh:
			return
		case processingResponse, ok = <-p.processingQueue:
			if !ok {
				return
			}
		}

		var r *eventData
		select {
		case <-stopCh:
			return
		case r, ok = <-processingResponse:
			if !ok {
				return
			}
		}

		select {
		case <-stopCh:
			return
		case p.resultChan <- r:
			if r.err != nil {
				return
			}
		}
	}
}

// Close closes the underlying r.
func (d *Decoder) Close() {
	d.decoder.Close()
	close(d.stopCh)
}
