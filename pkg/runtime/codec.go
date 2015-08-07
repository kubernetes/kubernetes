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
	"k8s.io/kubernetes/pkg/util/yaml"
)

// CodecFor returns a Codec that invokes Encode with the provided version.
func CodecFor(codec ObjectCodec, version string) Codec {
	return &codecWrapper{codec, version}
}

// yamlCodec converts YAML passed to the Decoder methods to JSON.
type yamlCodec struct {
	// a Codec for JSON
	Codec
}

// yamlCodec implements Codec
var _ Codec = yamlCodec{}

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
	bytes, err := codec.Encode(obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// codecWrapper implements encoding to an alternative
// default version for a scheme.
type codecWrapper struct {
	ObjectCodec
	version string
}

// Encode implements Codec
func (c *codecWrapper) Encode(obj Object) ([]byte, error) {
	return c.EncodeToVersion(obj, c.version)
}

// TODO: Make this behaviour default when we move everyone away from
// the unversioned types.
//
// func (c *codecWrapper) Decode(data []byte) (Object, error) {
// 	return c.DecodeToVersion(data, c.version)
// }
