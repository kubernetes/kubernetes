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
	"io"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// Encoder implements the json.Encoder interface for io.Writers that
// should serialize WatchEvent objects into JSON. It will encode any object
// registered in the supplied codec and return an error otherwies.
type Encoder struct {
	w       io.Writer
	encoder *json.Encoder
	codec   runtime.Codec
}

// NewEncoder creates an Encoder for the given writer and codec
func NewEncoder(w io.Writer, codec runtime.Codec) *Encoder {
	return &Encoder{
		w:       w,
		encoder: json.NewEncoder(w),
		codec:   codec,
	}
}

// Encode writes an event to the writer. Returns an error
// if the writer is closed or an object can't be encoded.
func (e *Encoder) Encode(event *watch.Event) error {
	obj, err := Object(e.codec, event)
	if err != nil {
		return err
	}
	return e.encoder.Encode(obj)
}
