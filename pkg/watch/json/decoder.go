/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package json

import (
	"encoding/json"
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// Decoder implements the watch.Decoder interface for io.ReadClosers that
// have contents which consist of a series of watchEvent objects encoded via JSON.
// It will decode any object registered in the supplied codec.
type Decoder struct {
	r       io.ReadCloser
	decoder *json.Decoder
	codec   runtime.Codec
}

// NewDecoder creates an Decoder for the given writer and codec.
func NewDecoder(r io.ReadCloser, codec runtime.Codec) *Decoder {
	return &Decoder{
		r:       r,
		decoder: json.NewDecoder(r),
		codec:   codec,
	}
}

// Decode blocks until it can return the next object in the writer. Returns an error
// if the writer is closed or an object can't be decoded.
func (d *Decoder) Decode() (watch.EventType, runtime.Object, error) {
	var got WatchEvent
	if err := d.decoder.Decode(&got); err != nil {
		return "", nil, err
	}
	switch got.Type {
	case watch.Added, watch.Modified, watch.Deleted, watch.Error:
	default:
		return "", nil, fmt.Errorf("got invalid watch event type: %v", got.Type)
	}

	obj, err := d.codec.Decode(got.Object.RawJSON)
	if err != nil {
		return "", nil, fmt.Errorf("unable to decode watch event: %v", err)
	}
	return got.Type, obj, nil
}

// Close closes the underlying r.
func (d *Decoder) Close() {
	d.r.Close()
}
