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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
)

const processEventConcurrency = 10

// Decoder implements the watch.Decoder interface for io.ReadClosers that
// have contents which consist of a series of watchEvent objects encoded
// with the given streaming decoder. The internal objects will be then
// decoded by the embedded decoder.
type Decoder struct {
	decoder         streaming.Decoder
	embeddedDecoder runtime.Decoder
}

// NewDecoder creates an Decoder for the given writer and codec.
func NewDecoder(decoder streaming.Decoder, embeddedDecoder runtime.Decoder) *Decoder {
	return &Decoder{
		decoder:         decoder,
		embeddedDecoder: embeddedDecoder,
	}
}

// Decode blocks until it can return the next object in the reader. Returns an error
// if the reader is closed or an object can't be decoded.
func (d *Decoder) Decode() (watch.EventType, runtime.Object, error) {
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

func (d *Decoder) ConcurrentDecode() <-chan *watch.EventData {
	p := concurrentOrderedEventProcessing{
		resultChan:      make(chan *watch.EventData, processEventConcurrency-1),
		processingQueue: make(chan chan *watch.EventData, processEventConcurrency-1),
		decoder:         d.decoder,
		embeddedDecoder: d.embeddedDecoder,
	}
	go func() {
		p.scheduleEventProcessing()
	}()
	go func() {
		p.collectEventProcessing()
	}()
	return p.resultChan
}

type concurrentOrderedEventProcessing struct {
	resultChan      chan *watch.EventData
	processingQueue chan chan *watch.EventData
	decoder         streaming.Decoder
	embeddedDecoder runtime.Decoder
}

func (p *concurrentOrderedEventProcessing) scheduleEventProcessing() {
	for {
		got, err := decodeWatchEvent(p.decoder)
		processingResponse := make(chan *watch.EventData, 1)
		p.processingQueue <- processingResponse
		go func(watchEvent *metav1.WatchEvent, response chan<- *watch.EventData) {
			pr := &watch.EventData{
				Action: watch.EventType(watchEvent.Type),
			}
			if err != nil {
				pr.Err = err
			} else {
				obj, err := runtime.Decode(p.embeddedDecoder, got.Object.Raw)
				if err != nil {
					pr.Err = fmt.Errorf("unable to decode watch event: %v", err)
				} else {
					pr.Object = obj
				}
			}
			response <- pr
		}(&got, processingResponse)
		if err != nil {
			// close decoder reader
			p.decoder.Close()
			return
		}
	}
}

func (p *concurrentOrderedEventProcessing) collectEventProcessing() {
	for {
		processingResponse, ok := <-p.processingQueue
		if !ok {
			return
		}
		r := <-processingResponse
		p.resultChan <- r
		if r.Err != nil {
			// after send error, close resultChan
			close(p.resultChan)
			return
		}
	}
}

// Close closes the underlying r.
func (d *Decoder) Close() {
	d.decoder.Close()
}
