/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"errors"
	"fmt"

	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
)

// MarshalToYaml marshals an object into yaml.
func MarshalToYaml(obj runtime.Object, gv schema.GroupVersion) ([]byte, error) {
	return MarshalToYamlForCodecs(obj, gv, clientsetscheme.Codecs)
}

// MarshalToYamlForCodecs marshals an object into yaml using the specified codec
// TODO: Is specifying the gv really needed here?
// TODO: Can we support json out of the box easily here?
func MarshalToYamlForCodecs(obj runtime.Object, gv schema.GroupVersion, codecs serializer.CodecFactory) ([]byte, error) {
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return []byte{}, fmt.Errorf("unsupported media type %q", mediaType)
	}

	encoder := codecs.EncoderForVersion(info.Serializer, gv)
	return runtime.Encode(encoder, obj)
}

// UnmarshalFromYaml unmarshals yaml into an object.
func UnmarshalFromYaml(buffer []byte, gv schema.GroupVersion) (runtime.Object, error) {
	return UnmarshalFromYamlForCodecs(buffer, gv, clientsetscheme.Codecs)
}

// UnmarshalFromYamlForCodecs unmarshals yaml into an object using the specified codec
// TODO: Is specifying the gv really needed here?
// TODO: Can we support json out of the box easily here?
func UnmarshalFromYamlForCodecs(buffer []byte, gv schema.GroupVersion, codecs serializer.CodecFactory) (runtime.Object, error) {
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}

	decoder := codecs.DecoderToVersion(info.Serializer, gv)
	return runtime.Decode(decoder, buffer)
}

// ExtractAPIVersionAndKindFromYAML extracts the APIVersion and Kind fields from YAML bytes
func ExtractAPIVersionAndKindFromYAML(b []byte) (string, string, error) {
	decoded, err := LoadYAML(b)
	if err != nil {
		return "", "", fmt.Errorf("unable to decode config from bytes: %v", err)
	}

	kindStr, ok := decoded["kind"].(string)
	if !ok || len(kindStr) == 0 {
		return "", "", fmt.Errorf("any config file must have the kind field set")
	}
	apiVersionStr, ok := decoded["apiVersion"].(string)
	if !ok || len(apiVersionStr) == 0 {
		return "", "", fmt.Errorf("any config file must have the apiVersion field set")
	}
	return apiVersionStr, kindStr, nil
}

// GroupVersionKindFromBytes parses the bytes and returns the gvk
// TODO: Find a better way to do this, invoking the API machinery directly without first loading the yaml manually
func GroupVersionKindFromBytes(b []byte, codecs serializer.CodecFactory) (schema.GroupVersionKind, error) {
	apiVersionStr, kindStr, err := ExtractAPIVersionAndKindFromYAML(b)
	if err != nil {
		return schema.EmptyObjectKind.GroupVersionKind(), err
	}

	gv, err := schema.ParseGroupVersion(apiVersionStr)
	if err != nil {
		return schema.EmptyObjectKind.GroupVersionKind(), fmt.Errorf("unable to parse apiVersion: %v", err)
	}
	return gv.WithKind(kindStr), nil
}

// LoadYAML is a small wrapper around go-yaml that ensures all nested structs are map[string]interface{} instead of map[interface{}]interface{}.
func LoadYAML(bytes []byte) (map[string]interface{}, error) {
	var decoded map[interface{}]interface{}
	if err := yaml.Unmarshal(bytes, &decoded); err != nil {
		return map[string]interface{}{}, fmt.Errorf("couldn't unmarshal YAML: %v", err)
	}

	converted, ok := convert(decoded).(map[string]interface{})
	if !ok {
		return map[string]interface{}{}, errors.New("yaml is not a map")
	}

	return converted, nil
}

// https://stackoverflow.com/questions/40737122/convert-yaml-to-json-without-struct-golang
func convert(i interface{}) interface{} {
	switch x := i.(type) {
	case map[interface{}]interface{}:
		m2 := map[string]interface{}{}
		for k, v := range x {
			m2[k.(string)] = convert(v)
		}
		return m2
	case []interface{}:
		for i, v := range x {
			x[i] = convert(v)
		}
	}
	return i
}
