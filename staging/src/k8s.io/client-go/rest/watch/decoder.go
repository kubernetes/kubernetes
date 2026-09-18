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
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
)

const watchObjectDecodeConcurrency = 10

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
	res, _, err := d.decoder.Decode(nil, &got)
	if err != nil {
		return "", nil, err
	}
	if res != &got {
		return "", nil, fmt.Errorf("unable to decode to metav1.WatchEvent")
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return "", nil, fmt.Errorf("got invalid watch event type: %v", got.Type)
	}

	obj, err := runtime.Decode(d.embeddedDecoder, got.Object.Raw)
	if err != nil {
		return "", nil, fmt.Errorf("unable to decode watch event: %v", err)
	}
	return watch.EventType(got.Type), obj, nil
}

// Close closes the underlying r.
func (d *Decoder) Close() {
	d.decoder.Close()
}

type decoderResult struct {
	action watch.EventType
	object runtime.Object
	err    error
}

type frameDecoder struct {
	framer            streaming.Framer
	watchEventDecoder runtime.Decoder
	embeddedDecoder   runtime.Decoder

	done            chan struct{}
	processingQueue chan chan decoderResult
	closeOnce       sync.Once
}

// NewFrameDecoder creates a Decoder that reads raw watch event frames serially
// and decodes each watch event frame concurrently while preserving order.
func NewFrameDecoder(reader io.ReadCloser, watchEventDecoder runtime.Decoder, embeddedDecoder runtime.Decoder) watch.Decoder {
	decoder := streaming.NewDecoder(reader, watchEventDecoder)
	framer, ok := decoder.(streaming.Framer)
	if !ok {
		return NewDecoder(decoder, embeddedDecoder)
	}
	d := &frameDecoder{
		framer:            framer,
		watchEventDecoder: watchEventDecoder,
		embeddedDecoder:   embeddedDecoder,
		done:              make(chan struct{}),
		processingQueue:   make(chan chan decoderResult, watchObjectDecodeConcurrency-1),
	}
	go d.schedule()
	return d
}

func (d *frameDecoder) Decode() (watch.EventType, runtime.Object, error) {
	select {
	case <-d.done:
		return "", nil, fmt.Errorf("watch decoder is closed")
	case response := <-d.processingQueue:
		select {
		case <-d.done:
			return "", nil, fmt.Errorf("watch decoder is closed")
		case result := <-response:
			return result.action, result.object, result.err
		}
	}
}

func (d *frameDecoder) Close() {
	d.closeOnce.Do(func() {
		close(d.done)
		d.framer.Close()
	})
}

func (d *frameDecoder) schedule() {
	for {
		frame, err := d.framer.ReadFrame()
		if err != nil {
			d.enqueueResult(decoderResult{err: err})
			return
		}

		response := make(chan decoderResult, 1)
		select {
		case <-d.done:
			frame.Release()
			return
		case d.processingQueue <- response:
		}
		go d.decodeFrame(frame, response)
	}
}

func (d *frameDecoder) decodeFrame(frame *streaming.Frame, response chan<- decoderResult) {
	defer frame.Release()
	result := d.decodeFrameToResult(frame)
	select {
	case <-d.done:
	case response <- result:
	}
}

func (d *frameDecoder) decodeFrameToResult(frame *streaming.Frame) decoderResult {
	var got metav1.WatchEvent
	res, _, err := d.watchEventDecoder.Decode(frame.Data(), nil, &got)
	if err != nil {
		return decoderResult{err: err}
	}
	if res != &got {
		return decoderResult{err: fmt.Errorf("unable to decode to metav1.WatchEvent")}
	}
	switch got.Type {
	case string(watch.Added), string(watch.Modified), string(watch.Deleted), string(watch.Error), string(watch.Bookmark):
	default:
		return decoderResult{err: fmt.Errorf("got invalid watch event type: %v", got.Type)}
	}

	obj, err := runtime.Decode(d.embeddedDecoder, got.Object.Raw)
	if err != nil {
		return decoderResult{err: fmt.Errorf("unable to decode watch event: %v", err)}
	}
	return decoderResult{action: watch.EventType(got.Type), object: obj}
}

func (d *frameDecoder) enqueueResult(result decoderResult) {
	response := make(chan decoderResult, 1)
	response <- result
	select {
	case <-d.done:
	case d.processingQueue <- response:
	}
}
