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
	RecognizesData(peek io.Reader) (bool, error)
}

func NewDecoder(decoders ...runtime.Decoder) runtime.Decoder {
	recognizing, blind := []RecognizingDecoder{}, []runtime.Decoder{}
	for _, d := range decoders {
		if r, ok := d.(RecognizingDecoder); ok {
			recognizing = append(recognizing, r)
		} else {
			blind = append(blind, d)
		}
	}
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
