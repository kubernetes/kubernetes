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
	"encoding/base64"
	"fmt"
	"io"
	"net/url"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion/queryparams"
)

// codec binds an encoder and decoder.
type codec struct {
	Encoder
	Decoder
}

// NewCodec creates a Codec from an Encoder and Decoder.
func NewCodec(e Encoder, d Decoder) Codec {
	return codec{e, d}
}

// Encode is a convenience wrapper for encoding to a []byte from an Encoder
func Encode(e Encoder, obj Object, overrides ...unversioned.GroupVersion) ([]byte, error) {
	// TODO: reuse buffer
	buf := &bytes.Buffer{}
	if err := e.EncodeToStream(obj, buf, overrides...); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Decode is a convenience wrapper for decoding data into an Object.
func Decode(d Decoder, data []byte) (Object, error) {
	obj, _, err := d.Decode(data, nil, nil)
	return obj, err
}

// DecodeInto performs a Decode into the provided object.
func DecodeInto(d Decoder, data []byte, into Object) error {
	out, gvk, err := d.Decode(data, nil, into)
	if err != nil {
		return err
	}
	if out != into {
		return fmt.Errorf("unable to decode %s into %v", gvk, reflect.TypeOf(into))
	}
	return nil
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(e Encoder, obj Object) string {
	bytes, err := Encode(e, obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// UseOrCreateObject returns obj if the canonical ObjectKind returned by the provided typer matches gvk, or
// invokes the ObjectCreator to instantiate a new gvk. Returns an error if the typer cannot find the object.
func UseOrCreateObject(t Typer, c ObjectCreater, gvk unversioned.GroupVersionKind, obj Object) (Object, error) {
	if obj != nil {
		into, _, err := t.ObjectKind(obj)
		if err != nil {
			return nil, err
		}
		if gvk == *into {
			return obj, nil
		}
	}
	return c.New(gvk)
}

// NoopEncoder converts an Decoder to a Serializer or Codec for code that expects them but only uses decoding.
type NoopEncoder struct {
	Decoder
}

var _ Serializer = NoopEncoder{}

func (n NoopEncoder) EncodeToStream(obj Object, w io.Writer, overrides ...unversioned.GroupVersion) error {
	return fmt.Errorf("encoding is not allowed for this codec: %v", reflect.TypeOf(n.Decoder))
}

// NoopDecoder converts an Encoder to a Serializer or Codec for code that expects them but only uses encoding.
type NoopDecoder struct {
	Encoder
}

var _ Serializer = NoopDecoder{}

func (n NoopDecoder) Decode(data []byte, gvk *unversioned.GroupVersionKind, into Object) (Object, *unversioned.GroupVersionKind, error) {
	return nil, nil, fmt.Errorf("decoding is not allowed for this codec: %v", reflect.TypeOf(n.Encoder))
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

// DecodeParameters converts the provided url.Values into an object of type From with the kind of into, and then
// converts that object to into (if necessary). Returns an error if the operation cannot be completed.
func (c *parameterCodec) DecodeParameters(parameters url.Values, from unversioned.GroupVersion, into Object) error {
	if len(parameters) == 0 {
		return nil
	}
	targetGVK, _, err := c.typer.ObjectKind(into)
	if err != nil {
		return err
	}
	if targetGVK.GroupVersion() == from {
		return c.convertor.Convert(&parameters, into)
	}
	input, err := c.creator.New(from.WithKind(targetGVK.Kind))
	if err != nil {
		return err
	}
	if err := c.convertor.Convert(&parameters, input); err != nil {
		return err
	}
	return c.convertor.Convert(input, into)
}

// EncodeParameters converts the provided object into the to version, then converts that object to url.Values.
// Returns an error if conversion is not possible.
func (c *parameterCodec) EncodeParameters(obj Object, to unversioned.GroupVersion) (url.Values, error) {
	gvk, _, err := c.typer.ObjectKind(obj)
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

type base64Serializer struct {
	Serializer
}

func NewBase64Serializer(s Serializer) Serializer {
	return &base64Serializer{s}
}

func (s base64Serializer) EncodeToStream(obj Object, stream io.Writer, overrides ...unversioned.GroupVersion) error {
	e := base64.NewEncoder(base64.StdEncoding, stream)
	err := s.Serializer.EncodeToStream(obj, e, overrides...)
	e.Close()
	return err
}

func (s base64Serializer) Decode(data []byte, defaults *unversioned.GroupVersionKind, into Object) (Object, *unversioned.GroupVersionKind, error) {
	out := make([]byte, base64.StdEncoding.DecodedLen(len(data)))
	n, err := base64.StdEncoding.Decode(out, data)
	if err != nil {
		return nil, nil, err
	}
	return s.Serializer.Decode(out[:n], defaults, into)
}
