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

package runtime

import (
	"bytes"
	"fmt"
	"io"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(codec Serializer, obj Object) string {
	bytes, err := Encode(codec, obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// Encode is a convenience wrapper for encoding to a []byte from a Serializer
func Encode(s Serializer, obj Object) ([]byte, error) {
	// TODO: reuse buffer
	buf := &bytes.Buffer{}
	if err := s.EncodeToStream(obj, buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// NoopEncoder converts an Decoder to a Serializer or Codec for code that expects them but only uses decoding.
type NoopEncoder struct {
	Decoder
}

func (n NoopEncoder) EncodeToStream(obj Object, w io.Writer) error {
	return fmt.Errorf("encoding is not allowed for this codec: %v", reflect.TypeOf(n.Decoder))
}

// NoopDecoder converts an Encoder to a Serializer or Codec for code that expects them but only uses encoding.
type NoopDecoder struct {
	Encoder
}

func (n NoopDecoder) Decode(data []byte, gvk *unversioned.GroupVersionKind) (Object, *unversioned.GroupVersionKind, error) {
	return nil, nil, fmt.Errorf("decoding is not allowed for this codec: %v", reflect.TypeOf(n.Encoder))
}
