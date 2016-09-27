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

package runtime

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/url"
	"reflect"

	"k8s.io/client-go/1.5/pkg/api/unversioned"
	"k8s.io/client-go/1.5/pkg/conversion/queryparams"
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
func Encode(e Encoder, obj Object) ([]byte, error) {
	// TODO: reuse buffer
	buf := &bytes.Buffer{}
	if err := e.Encode(obj, buf); err != nil {
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
func UseOrCreateObject(t ObjectTyper, c ObjectCreater, gvk unversioned.GroupVersionKind, obj Object) (Object, error) {
	if obj != nil {
		kinds, _, err := t.ObjectKinds(obj)
		if err != nil {
			return nil, err
		}
		for _, kind := range kinds {
			if gvk == kind {
				return obj, nil
			}
		}
	}
	return c.New(gvk)
}

// NoopEncoder converts an Decoder to a Serializer or Codec for code that expects them but only uses decoding.
type NoopEncoder struct {
	Decoder
}

var _ Serializer = NoopEncoder{}

func (n NoopEncoder) Encode(obj Object, w io.Writer) error {
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
		typer:     scheme,
		convertor: scheme,
		creator:   scheme,
	}
}

// parameterCodec implements conversion to and from query parameters and objects.
type parameterCodec struct {
	typer     ObjectTyper
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
	targetGVKs, _, err := c.typer.ObjectKinds(into)
	if err != nil {
		return err
	}
	targetGVK := targetGVKs[0]
	if targetGVK.GroupVersion() == from {
		return c.convertor.Convert(&parameters, into, nil)
	}
	input, err := c.creator.New(from.WithKind(targetGVK.Kind))
	if err != nil {
		return err
	}
	if err := c.convertor.Convert(&parameters, input, nil); err != nil {
		return err
	}
	return c.convertor.Convert(input, into, nil)
}

// EncodeParameters converts the provided object into the to version, then converts that object to url.Values.
// Returns an error if conversion is not possible.
func (c *parameterCodec) EncodeParameters(obj Object, to unversioned.GroupVersion) (url.Values, error) {
	gvks, _, err := c.typer.ObjectKinds(obj)
	if err != nil {
		return nil, err
	}
	gvk := gvks[0]
	if to != gvk.GroupVersion() {
		out, err := c.convertor.ConvertToVersion(obj, to)
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

func (s base64Serializer) Encode(obj Object, stream io.Writer) error {
	e := base64.NewEncoder(base64.StdEncoding, stream)
	err := s.Serializer.Encode(obj, e)
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

var (
	// InternalGroupVersioner will always prefer the internal version for a given group version kind.
	InternalGroupVersioner GroupVersioner = internalGroupVersioner{}
	// DisabledGroupVersioner will reject all kinds passed to it.
	DisabledGroupVersioner GroupVersioner = disabledGroupVersioner{}
)

type internalGroupVersioner struct{}

// KindForGroupVersionKinds returns an internal Kind if one is found, or converts the first provided kind to the internal version.
func (internalGroupVersioner) KindForGroupVersionKinds(kinds []unversioned.GroupVersionKind) (unversioned.GroupVersionKind, bool) {
	for _, kind := range kinds {
		if kind.Version == APIVersionInternal {
			return kind, true
		}
	}
	for _, kind := range kinds {
		return unversioned.GroupVersionKind{Group: kind.Group, Version: APIVersionInternal, Kind: kind.Kind}, true
	}
	return unversioned.GroupVersionKind{}, false
}

type disabledGroupVersioner struct{}

// KindForGroupVersionKinds returns false for any input.
func (disabledGroupVersioner) KindForGroupVersionKinds(kinds []unversioned.GroupVersionKind) (unversioned.GroupVersionKind, bool) {
	return unversioned.GroupVersionKind{}, false
}

// GroupVersioners implements GroupVersioner and resolves to the first exact match for any kind.
type GroupVersioners []GroupVersioner

// KindForGroupVersionKinds returns the first match of any of the group versioners, or false if no match occured.
func (gvs GroupVersioners) KindForGroupVersionKinds(kinds []unversioned.GroupVersionKind) (unversioned.GroupVersionKind, bool) {
	for _, gv := range gvs {
		target, ok := gv.KindForGroupVersionKinds(kinds)
		if !ok {
			continue
		}
		return target, true
	}
	return unversioned.GroupVersionKind{}, false
}

// Assert that unversioned.GroupVersion and GroupVersions implement GroupVersioner
var _ GroupVersioner = unversioned.GroupVersion{}
var _ GroupVersioner = unversioned.GroupVersions{}
var _ GroupVersioner = multiGroupVersioner{}

type multiGroupVersioner struct {
	target             unversioned.GroupVersion
	acceptedGroupKinds []unversioned.GroupKind
}

// NewMultiGroupVersioner returns the provided group version for any kind that matches one of the provided group kinds.
// Kind may be empty in the provided group kind, in which case any kind will match.
func NewMultiGroupVersioner(gv unversioned.GroupVersion, groupKinds ...unversioned.GroupKind) GroupVersioner {
	if len(groupKinds) == 0 || (len(groupKinds) == 1 && groupKinds[0].Group == gv.Group) {
		return gv
	}
	return multiGroupVersioner{target: gv, acceptedGroupKinds: groupKinds}
}

// KindForGroupVersionKinds returns the target group version if any kind matches any of the original group kinds. It will
// use the originating kind where possible.
func (v multiGroupVersioner) KindForGroupVersionKinds(kinds []unversioned.GroupVersionKind) (unversioned.GroupVersionKind, bool) {
	for _, src := range kinds {
		for _, kind := range v.acceptedGroupKinds {
			if kind.Group != src.Group {
				continue
			}
			if len(kind.Kind) > 0 && kind.Kind != src.Kind {
				continue
			}
			return v.target.WithKind(src.Kind), true
		}
	}
	return unversioned.GroupVersionKind{}, false
}
