/*
Copyright 2014 Google Inc. All rights reserved.

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

package tools

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// APIEventDecoder implements the watch.Decoder interface for io.ReadClosers that
// have contents which consist of a series of api.WatchEvent objects encoded via JSON.
type APIEventDecoder struct {
	stream  io.ReadCloser
	decoder *json.Decoder
}

// NewAPIEventDecoder makes an APIEventDecoder for the given stream.
func NewAPIEventDecoder(stream io.ReadCloser) *APIEventDecoder {
	return &APIEventDecoder{
		stream:  stream,
		decoder: json.NewDecoder(stream),
	}
}

// Decode blocks until it can return the next object in the stream. Returns an error
// if the stream is closed or an object can't be decoded.
func (d *APIEventDecoder) Decode() (action watch.EventType, object interface{}, err error) {
	var got api.WatchEvent
	err = d.decoder.Decode(&got)
	if err != nil {
		return action, nil, err
	}
	switch got.Type {
	case watch.Added, watch.Modified, watch.Deleted:
		return got.Type, got.Object.Object, err
	}
	return action, nil, fmt.Errorf("got invalid watch event type: %v", got.Type)
}

// Close closes the underlying stream.
func (d *APIEventDecoder) Close() {
	d.stream.Close()
}
