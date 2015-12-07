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
	"net/url"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion/queryparams"
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
func Encode(s Encoder, obj Object) ([]byte, error) {
	// TODO: reuse buffer
	buf := &bytes.Buffer{}
	if err := s.EncodeToStream(obj, buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func UseOrCreateObject(t Typer, c ObjectCreater, gvk unversioned.GroupVersionKind, obj Object) (Object, error) {
	if obj != nil {
		into, err := t.ObjectVersionAndKind(obj)
		if err != nil {
			return nil, err
		}
		if gvk == *into {
			return obj, nil
		}
	}
	return c.New(gvk.GroupVersion().String(), gvk.Kind)
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

// DecodeInto performs a Decode and returns an error if the output object is not the same as the into object.
func DecodeInto(d Decoder, data []byte, gvk *unversioned.GroupVersionKind, into Object) (*unversioned.GroupVersionKind, error) {
	out, gvk, err := d.Decode(data, gvk, into)
	if err != nil {
		return gvk, err
	}
	if out != into {
		return gvk, fmt.Errorf("unable to decode %s into %v", gvk.String, reflect.TypeOf(into))
	}
	return gvk, nil
}

// NewParameterCodec creates a ParameterCodec capable of transforming url values into versioned objects and back.
func NewParameterCodec(scheme *Scheme) ParameterCodec {
	return &parameterCodec{
		typer:     ObjectTyperToTyper(scheme),
		convertor: scheme,
		creator:   scheme,
	}
}

// parameterCodec implements conversion to and from query parameters and objects.
type parameterCodec struct {
	typer     Typer
	convertor ObjectConvertor
	creator   ObjectCreater
}

var _ ParameterCodec = &parameterCodec{}

func (c *parameterCodec) DecodeParameters(parameters url.Values, from unversioned.GroupVersion, into Object) error {
	if len(parameters) == 0 {
		return nil
	}
	gvk, err := c.typer.ObjectVersionAndKind(into)
	if err != nil {
		return err
	}
	if gvk.GroupVersion() == from {
		return c.convertor.Convert(&parameters, into)
	}
	input, err := c.creator.New(from.String(), gvk.Kind)
	if err != nil {
		return err
	}
	if err := c.convertor.Convert(&parameters, input); err != nil {
		return err
	}
	return c.convertor.Convert(input, into)
}

func (c *parameterCodec) EncodeParameters(obj Object, to unversioned.GroupVersion) (url.Values, error) {
	gvk, err := c.typer.ObjectVersionAndKind(obj)
	if err != nil {
		return nil, err
	}
	if to != gvk.GroupVersion() {
		out, err := c.convertor.ConvertToVersion(obj, to.String())
		if err != nil {
			return nil, err
		}
		obj = out
	}
	return queryparams.Convert(obj)
}
