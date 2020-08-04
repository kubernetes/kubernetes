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

package recognizer

import (
	"bufio"
	"bytes"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type RecognizingDecoder interface {
	runtime.Decoder
	// RecognizesData should return true if the input provided in the provided reader
	// belongs to this decoder, or an error if the data could not be read or is ambiguous.
	// Unknown is true if the data could not be determined to match the decoder type.
	// Decoders should assume that they can read as much of peek as they need (as the caller
	// provides) and may return unknown if the data provided is not sufficient to make a
	// a determination. When peek returns EOF that may mean the end of the input or the
	// end of buffered input - recognizers should return the best guess at that time.
	RecognizesData(peek io.Reader) (ok, unknown bool, err error)
}

// NewDecoder creates a decoder that will attempt multiple decoders in an order defined
// by:
//
// 1. The decoder implements RecognizingDecoder and identifies the data
// 2. All other decoders, and any decoder that returned true for unknown.
//
// The order passed to the constructor is preserved within those priorities.
func NewDecoder(decoders ...runtime.Decoder) runtime.Decoder {
	return &decoder{
		decoders: decoders,
	}
}

type decoder struct {
	decoders []runtime.Decoder
}

var _ RecognizingDecoder = &decoder{}

func (d *decoder) RecognizesData(peek io.Reader) (bool, bool, error) {
	var (
		lastErr    error
		anyUnknown bool
	)
	data, _ := bufio.NewReaderSize(peek, 1024).Peek(1024)
	for _, r := range d.decoders {
		switch t := r.(type) {
		case RecognizingDecoder:
			ok, unknown, err := t.RecognizesData(bytes.NewBuffer(data))
			if err != nil {
				lastErr = err
				continue
			}
			anyUnknown = anyUnknown || unknown
			if !ok {
				continue
			}
			return true, false, nil
		}
	}
	return false, anyUnknown, lastErr
}

func (d *decoder) Decode(data []byte, gvk *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	var (
		lastErr error
		skipped []runtime.Decoder
	)

	// try recognizers, record any decoders we need to give a chance later
	for _, r := range d.decoders {
		switch t := r.(type) {
		case RecognizingDecoder:
			buf := bytes.NewBuffer(data)
			ok, unknown, err := t.RecognizesData(buf)
			if err != nil {
				lastErr = err
				continue
			}
			if unknown {
				skipped = append(skipped, t)
				continue
			}
			if !ok {
				continue
			}
			return r.Decode(data, gvk, into)
		default:
			skipped = append(skipped, t)
		}
	}

	// try recognizers that returned unknown or didn't recognize their data
	for _, r := range skipped {
		out, actual, err := r.Decode(data, gvk, into)
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
