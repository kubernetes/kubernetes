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

package versioned

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/streaming"
	"k8s.io/kubernetes/pkg/watch"
)

// Encoder serializes watch.Events into io.Writer. The internal objects
// are encoded using embedded encoder, and the outer Event is serialized
// using encoder.
type Encoder struct {
	encoder         streaming.Encoder
	embeddedEncoder runtime.Encoder
}

func NewEncoder(encoder streaming.Encoder, embeddedEncoder runtime.Encoder) *Encoder {
	return &Encoder{
		encoder:         encoder,
		embeddedEncoder: embeddedEncoder,
	}
}

// Encode writes an event to the writer. Returns an error
// if the writer is closed or an object can't be encoded.
func (e *Encoder) Encode(event *watch.Event) error {
	data, err := runtime.Encode(e.embeddedEncoder, event.Object)
	if err != nil {
		return err
	}
	// FIXME: get rid of json.RawMessage.
	return e.encoder.Encode(&Event{string(event.Type), runtime.RawExtension{Raw: json.RawMessage(data)}})
}
