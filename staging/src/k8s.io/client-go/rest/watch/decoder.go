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

// TimingReadCloser wraps an io.ReadCloser and accumulates time spent in Read().
type TimingReadCloser struct {
	io.ReadCloser
	TotalReadTime time.Duration
}

func (t *TimingReadCloser) Read(p []byte) (int, error) {
	start := time.Now()
	n, err := t.ReadCloser.Read(p)
	t.TotalReadTime += time.Since(start)
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

// NewDecoderWithNetworkTiming creates a Decoder that tracks network read time separately.
func NewDecoderWithNetworkTiming(decoder streaming.Decoder, embeddedDecoder runtime.Decoder, networkReader *TimingReadCloser) *Decoder {
	return &Decoder{
		decoder:         decoder,
		embeddedDecoder: embeddedDecoder,
		networkReader:   networkReader,
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

// NetworkTiming implements watch.DecoderNetworkTimingReporter.
func (d *Decoder) NetworkTiming() time.Duration {
	if d.networkReader == nil {
		return 0
	}
	return d.networkReader.TotalReadTime
}

// Close closes the underlying r.
func (d *Decoder) Close() {
	d.decoder.Close()
}
