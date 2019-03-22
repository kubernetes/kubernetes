/*
Copyright 2019 The Kubernetes Authors.

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

package serializer

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

// EncodingFormat is a high-level, typed constant configuring how an object should be encoded.
type EncodingFormat string

const (
	// ContentTypeYAML encodes a config file in the YAML format.
	ContentTypeYAML EncodingFormat = "application/yaml"
	// ContentTypeJSON encodes a config file in compact JSON format.
	ContentTypeJSON EncodingFormat = "application/json"
)

// NewStrictYAMLJSONSerializer creates a new implementation of the StrictYAMLJSONSerializer interface.
func NewStrictYAMLJSONSerializer(scheme *runtime.Scheme, codecs *serializer.CodecFactory) StrictYAMLJSONSerializer {
	knownEncoders := map[EncodingFormat]runtime.Encoder{}
	for _, serializerInfo := range codecs.SupportedMediaTypes() {
		switch serializerInfo.MediaType {
		case "application/json":
			knownEncoders[ContentTypeJSON] = serializerInfo.Serializer
		case "application/yaml":
			knownEncoders[ContentTypeYAML] = serializerInfo.Serializer
		}
	}

	// Create the strict deserializer, and creates a decoder later.
	strictYAMLSerializer := json.NewStrictYAMLSerializer(json.DefaultMetaFactory, scheme, scheme)
	universalDecoder := codecs.CodecForVersions(nil, strictYAMLSerializer, nil, runtime.InternalGroupVersioner)

	return &strictYAMLJSONSerializer{scheme, codecs, knownEncoders, universalDecoder}
}

// StrictYAMLJSONSerializer provides encoding/decoding for a configuration file.
type StrictYAMLJSONSerializer interface {
	DecodeInto([]byte, runtime.Object) error
	Encode(EncodingFormat, schema.GroupVersion, runtime.Object) ([]byte, error)
}

// strictYAMLJSONSerializer implements the StrictYAMLJSONSerializer interface.
var _ StrictYAMLJSONSerializer = &strictYAMLJSONSerializer{}

type strictYAMLJSONSerializer struct {
	scheme        *runtime.Scheme
	codecs        *serializer.CodecFactory
	knownEncoders map[EncodingFormat]runtime.Encoder
	decoder       runtime.Decoder
}

// DecodeInto decodes bytes into a pointer of the desired type.
func (cs *strictYAMLJSONSerializer) DecodeInto(data []byte, into runtime.Object) error {
	out, gvk, err := cs.decoder.Decode(data, nil, into)
	// return a structured error if the group was registered with the scheme but the version was unrecognized.
	if gvk != nil && err != nil {
		if cs.scheme.IsGroupRegistered(gvk.Group) && !cs.scheme.IsVersionRegistered(gvk.GroupVersion()) {
			return NewUnrecognizedVersionError(*gvk, data)
		}
	}
	if err != nil {
		return err
	}
	if out != into {
		return fmt.Errorf("unable to decode %s into %T", gvk, reflect.TypeOf(into))
	}
	return nil
}

// Encode encodes the object into a byte slice for the specific format and version.
func (cs *strictYAMLJSONSerializer) Encode(format EncodingFormat, gv schema.GroupVersion, obj runtime.Object) ([]byte, error) {
	knownEncoder, ok := cs.knownEncoders[format]
	if !ok {
		return []byte{}, fmt.Errorf("encoding format not supported: %s", format)
	}
	// versionSpecificEncoder writes out the specific format bytes for exactly this group version.
	versionSpecificEncoder := cs.codecs.EncoderForVersion(knownEncoder, gv)
	// Encode the object to the specific format for the given version.
	return runtime.Encode(versionSpecificEncoder, obj)
}
