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

// EncodingFormat is a high-level, typed constant configuring how an object should be encoded
type EncodingFormat string

const (
	// YAML encodes a config file in the YAML format
	YAML EncodingFormat = "YAML"
	// JSON encodes a config file in compact JSON format
	JSON EncodingFormat = "JSON"
)

// NewConfigSerializer creates a new implementation of the ConfigSerializer interface
func NewConfigSerializer(scheme *runtime.Scheme, codecs *serializer.CodecFactory) ConfigSerializer {
	// Store a map of known encoders, indexed by EncodingFormat
	encoderMap := map[EncodingFormat]runtime.Encoder{}
	for _, serializerInfo := range codecs.SupportedMediaTypes() {
		switch serializerInfo.MediaType {
		case "application/json":
			encoderMap[JSON] = serializerInfo.Serializer
		case "application/yaml":
			encoderMap[YAML] = serializerInfo.Serializer
		}
	}

	// Create the strict deserializer, and later the
	yamlStrictSerializer := json.NewStrictYAMLSerializer(json.DefaultMetaFactory, scheme, scheme)
	universalDecoder := codecs.CodecForVersions(nil, yamlStrictSerializer, nil, runtime.InternalGroupVersioner)

	return &configSerializer{scheme, codecs, encoderMap, universalDecoder}
}

// ConfigSerializer provides encoding/decoding for a configuration file
type ConfigSerializer interface {
	DecodeInto([]byte, runtime.Object) error
	Encode(EncodingFormat, schema.GroupVersion, runtime.Object) ([]byte, error)
}

// configSerializer implements the ConfigSerializer interface
var _ ConfigSerializer = &configSerializer{}

type configSerializer struct {
	scheme     *runtime.Scheme
	codecs     *serializer.CodecFactory
	encoderMap map[EncodingFormat]runtime.Encoder
	decoder    runtime.Decoder
}

// DecodeInto decodes bytes into a pointer of the desired type
func (cs *configSerializer) DecodeInto(data []byte, into runtime.Object) error {
	out, gvk, err := cs.decoder.Decode(data, nil, into)
	// Return a structured error if the group was registered with the scheme but the version was unrecognized
	if gvk != nil && err != nil {
		if cs.scheme.IsGroupRegistered(gvk.Group) && !cs.scheme.IsVersionRegistered(gvk.GroupVersion()) {
			return NewUnrecognizedVersionError("please specify a correct API version", *gvk, data)
		}
	}
	if err != nil {
		return err
	}
	if out != into {
		return fmt.Errorf("unable to decode %s into %v", gvk, reflect.TypeOf(into))
	}
	return nil
}

// Encode encodes the object into a byte slice for the specific format and version
func (cs *configSerializer) Encode(format EncodingFormat, gv schema.GroupVersion, obj runtime.Object) ([]byte, error) {
	encoder, ok := cs.encoderMap[format]
	if !ok {
		return []byte{}, fmt.Errorf("encoding format not supported: %s", format)
	}
	// versionSpecificEncoder writes out YAML bytes for exactly this group version
	versionSpecificEncoder := cs.codecs.EncoderForVersion(encoder, gv)
	// Encode the object to YAML for the given version
	return runtime.Encode(versionSpecificEncoder, obj)
}
