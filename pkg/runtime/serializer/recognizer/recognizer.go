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

package recognizer

import (
	"bytes"
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type RecognizingDecoder interface {
	runtime.Decoder
	// AccurateRecognizer should be true if this decoder can unambigously determine
	// whether the content is known to it. Allows RecognizingDecoders to adjust certainty.
	AccurateRecognizer() bool
	// RecognizesData should return true if the input provided in the provided reader
	// belongs to this decoder, or an error if the data could not be read or is ambiguous.
	RecognizesData(peek io.Reader) (bool, error)
}

// NewDecoder creates a decoder that will attempt multiple decoders in an order defined
// by:
//
// 1. The decoder implements RecognizingDecoder and returns true for AccurateRecognizer
// 2. The decoder implements RecognizingDecoder and returns false for AccurateRecognizer
// 3. All other decoders
//
// The order passed to the constructor is preserved within those priorities. Only the
// last error encountered will be returned.
func NewDecoder(decoders ...runtime.Decoder) runtime.Decoder {
	recognizing, blind := []RecognizingDecoder{}, []runtime.Decoder{}
	var ambiguous []runtime.Decoder
	for _, d := range decoders {
		if r, ok := d.(RecognizingDecoder); ok {
			if r.AccurateRecognizer() {
				recognizing = append(recognizing, r)
			} else {
				ambiguous = append(ambiguous, r)
			}
		} else {
			blind = append(blind, d)
		}
	}
	blind = append(ambiguous, blind...)
	return &decoder{
		recognizing: recognizing,
		blind:       blind,
	}
}

type decoder struct {
	recognizing []RecognizingDecoder
	blind       []runtime.Decoder
}

func (d *decoder) Decode(data []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	var lastErr error
	for _, r := range d.recognizing {
		buf := bytes.NewBuffer(data)
		ok, err := r.RecognizesData(buf)
		if err != nil {
			lastErr = err
			continue
		}
		if !ok {
			continue
		}
		return r.Decode(data, gvk, into)
	}
	for _, d := range d.blind {
		out, actual, err := d.Decode(data, gvk, into)
		if err != nil {
			lastErr = err
			continue
		}
		return out, actual, nil
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no serialization format matched the provided data")
	}
	return nil, nil, lastErr
}
