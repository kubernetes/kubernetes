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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
)

// TimingReadCloser measures the time spent reading a watch response.
type TimingReadCloser struct {
	io.ReadCloser
	readTime time.Duration
}

func (r *TimingReadCloser) Read(p []byte) (int, error) {
	start := time.Now()
	n, err := r.ReadCloser.Read(p)
	r.readTime += time.Since(start)
	return n, err
}

// Decoder implements the watch.Decoder interface for io.ReadClosers that
// have contents which consist of a series of watchEvent objects encoded
// with the given streaming decoder. The internal objects will be then
// decoded by the embedded decoder.
type Decoder struct {
	decoder         streaming.Decoder
	embeddedDecoder runtime.Decoder
	networkReader   *TimingReadCloser
}

// NewDecoder creates an Decoder for the given writer and codec.
func NewDecoder(decoder streaming.Decoder, embeddedDecoder runtime.Decoder) *Decoder {
	return &Decoder{
		decoder:         decoder,
		embeddedDecoder: embeddedDecoder,
	}
}

// NewDecoderWithNetworkTiming creates a decoder that can report watch response read time.
func NewDecoderWithNetworkTiming(decoder streaming.Decoder, embeddedDecoder runtime.Decoder, reader *TimingReadCloser) *Decoder {
	return &Decoder{decoder: decoder, embeddedDecoder: embeddedDecoder, networkReader: reader}
}

// NetworkTiming reports the accumulated time spent reading the watch response.
func (d *Decoder) NetworkTiming() time.Duration {
	if d.networkReader == nil {
		return 0
	}
	return d.networkReader.readTime
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
