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
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/yaml"
)

// Encode is a convenience wrapper for encoding to a []byte from an Encoder
// TODO: these are transitional interfaces to reduce refactor cost as Codec is altered.
func Encode(e Encoder, obj Object) ([]byte, error) {
	return e.Encode(obj)
}

// Decode is a convenience wrapper for decoding data into an Object.
// TODO: these are transitional interfaces to reduce refactor cost as Codec is altered.
func Decode(d Decoder, data []byte) (Object, error) {
	return d.Decode(data)
}

// DecodeInto performs a Decode into the provided object.
// TODO: these are transitional interfaces to reduce refactor cost as Codec is altered.
func DecodeInto(d Decoder, data []byte, into Object) error {
	return d.DecodeInto(data, into)
}

// CodecFor returns a Codec that invokes Encode with the provided version.
func CodecFor(codec ObjectCodec, version unversioned.GroupVersion) Codec {
	return &codecWrapper{codec, version}
}

// yamlCodec converts YAML passed to the Decoder methods to JSON.
type yamlCodec struct {
	// a Codec for JSON
	Codec
}

// yamlCodec implements Codec
var _ Codec = yamlCodec{}
var _ Decoder = yamlCodec{}

// YAMLDecoder adds YAML decoding support to a codec that supports JSON.
func YAMLDecoder(codec Codec) Codec {
	return &yamlCodec{codec}
}

func (c yamlCodec) Decode(data []byte) (Object, error) {
	out, err := yaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	data = out
	return c.Codec.Decode(data)
}

func (c yamlCodec) DecodeInto(data []byte, obj Object) error {
	out, err := yaml.ToJSON(data)
	if err != nil {
		return err
	}
	data = out
	return c.Codec.DecodeInto(data, obj)
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(codec Codec, obj Object) string {
	bytes, err := Encode(codec, obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// codecWrapper implements encoding to an alternative
// default version for a scheme.
type codecWrapper struct {
	ObjectCodec
	version unversioned.GroupVersion
}

// codecWrapper implements Decoder
var _ Decoder = &codecWrapper{}

// Encode implements Codec
func (c *codecWrapper) Encode(obj Object) ([]byte, error) {
	return c.EncodeToVersion(obj, c.version.String())
}

func (c *codecWrapper) EncodeToStream(obj Object, stream io.Writer) error {
	return c.EncodeToVersionStream(obj, c.version.String(), stream)
}

// TODO: Make this behaviour default when we move everyone away from
// the unversioned types.
//
// func (c *codecWrapper) Decode(data []byte) (Object, error) {
// 	return c.DecodeToVersion(data, c.version)
// }
