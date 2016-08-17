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

package json

import (
	"encoding/json"
	"io"

	"github.com/ghodss/yaml"
	"github.com/ugorji/go/codec"

	"k8s.io/client-go/1.4/pkg/api/unversioned"
	"k8s.io/client-go/1.4/pkg/runtime"
	"k8s.io/client-go/1.4/pkg/runtime/serializer/recognizer"
	"k8s.io/client-go/1.4/pkg/util/framer"
	utilyaml "k8s.io/client-go/1.4/pkg/util/yaml"
)

// NewSerializer creates a JSON serializer that handles encoding versioned objects into the proper JSON form. If typer
// is not nil, the object has the group, version, and kind fields set.
func NewSerializer(meta MetaFactory, creater runtime.ObjectCreater, typer runtime.ObjectTyper, pretty bool) *Serializer {
	return &Serializer{
		meta:    meta,
		creater: creater,
		typer:   typer,
		yaml:    false,
		pretty:  pretty,
	}
}

// NewYAMLSerializer creates a YAML serializer that handles encoding versioned objects into the proper YAML form. If typer
// is not nil, the object has the group, version, and kind fields set. This serializer supports only the subset of YAML that
// matches JSON, and will error if constructs are used that do not serialize to JSON.
func NewYAMLSerializer(meta MetaFactory, creater runtime.ObjectCreater, typer runtime.ObjectTyper) *Serializer {
	return &Serializer{
		meta:    meta,
		creater: creater,
		typer:   typer,
		yaml:    true,
	}
}

type Serializer struct {
	meta    MetaFactory
	creater runtime.ObjectCreater
	typer   runtime.ObjectTyper
	yaml    bool
	pretty  bool
}

// Serializer implements Serializer
var _ runtime.Serializer = &Serializer{}
var _ recognizer.RecognizingDecoder = &Serializer{}

// Decode attempts to convert the provided data into YAML or JSON, extract the stored schema kind, apply the provided default gvk, and then
// load that data into an object matching the desired schema kind or the provided into. If into is *runtime.Unknown, the raw data will be
// extracted and no decoding will be performed. If into is not registered with the typer, then the object will be straight decoded using
// normal JSON/YAML unmarshalling. If into is provided and the original data is not fully qualified with kind/version/group, the type of
// the into will be used to alter the returned gvk. On success or most errors, the method will return the calculated schema kind.
func (s *Serializer) Decode(originalData []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	if versioned, ok := into.(*runtime.VersionedObjects); ok {
		into = versioned.Last()
		obj, actual, err := s.Decode(originalData, gvk, into)
		if err != nil {
			return nil, actual, err
		}
		versioned.Objects = []runtime.Object{obj}
		return versioned, actual, nil
	}

	data := originalData
	if s.yaml {
		altered, err := yaml.YAMLToJSON(data)
		if err != nil {
			return nil, nil, err
		}
		data = altered
	}

	actual, err := s.meta.Interpret(data)
	if err != nil {
		return nil, nil, err
	}

	if gvk != nil {
		// apply kind and version defaulting from provided default
		if len(actual.Kind) == 0 {
			actual.Kind = gvk.Kind
		}
		if len(actual.Version) == 0 && len(actual.Group) == 0 {
			actual.Group = gvk.Group
			actual.Version = gvk.Version
		}
		if len(actual.Version) == 0 && actual.Group == gvk.Group {
			actual.Version = gvk.Version
		}
	}

	if unk, ok := into.(*runtime.Unknown); ok && unk != nil {
		unk.Raw = originalData
		unk.ContentType = runtime.ContentTypeJSON
		unk.GetObjectKind().SetGroupVersionKind(*actual)
		return unk, actual, nil
	}

	if into != nil {
		types, _, err := s.typer.ObjectKinds(into)
		switch {
		case runtime.IsNotRegisteredError(err):
			if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(into); err != nil {
				return nil, actual, err
			}
			return into, actual, nil
		case err != nil:
			return nil, actual, err
		default:
			typed := types[0]
			if len(actual.Kind) == 0 {
				actual.Kind = typed.Kind
			}
			if len(actual.Version) == 0 && len(actual.Group) == 0 {
				actual.Group = typed.Group
				actual.Version = typed.Version
			}
			if len(actual.Version) == 0 && actual.Group == typed.Group {
				actual.Version = typed.Version
			}
		}
	}

	if len(actual.Kind) == 0 {
		return nil, actual, runtime.NewMissingKindErr(string(originalData))
	}
	if len(actual.Version) == 0 {
		return nil, actual, runtime.NewMissingVersionErr(string(originalData))
	}

	// use the target if necessary
	obj, err := runtime.UseOrCreateObject(s.typer, s.creater, *actual, into)
	if err != nil {
		return nil, actual, err
	}

	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(obj); err != nil {
		return nil, actual, err
	}
	return obj, actual, nil
}

// Encode serializes the provided object to the given writer.
func (s *Serializer) Encode(obj runtime.Object, w io.Writer) error {
	if s.yaml {
		json, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		data, err := yaml.JSONToYAML(json)
		if err != nil {
			return err
		}
		_, err = w.Write(data)
		return err
	}

	if s.pretty {
		data, err := json.MarshalIndent(obj, "", "  ")
		if err != nil {
			return err
		}
		_, err = w.Write(data)
		return err
	}
	encoder := json.NewEncoder(w)
	return encoder.Encode(obj)
}

// RecognizesData implements the RecognizingDecoder interface.
func (s *Serializer) RecognizesData(peek io.Reader) (ok, unknown bool, err error) {
	if s.yaml {
		// we could potentially look for '---'
		return false, true, nil
	}
	_, ok = utilyaml.GuessJSONStream(peek, 2048)
	return ok, false, nil
}

// Framer is the default JSON framing behavior, with newlines delimiting individual objects.
var Framer = jsonFramer{}

type jsonFramer struct{}

// NewFrameWriter implements stream framing for this serializer
func (jsonFramer) NewFrameWriter(w io.Writer) io.Writer {
	// we can write JSON objects directly to the writer, because they are self-framing
	return w
}

// NewFrameReader implements stream framing for this serializer
func (jsonFramer) NewFrameReader(r io.ReadCloser) io.ReadCloser {
	// we need to extract the JSON chunks of data to pass to Decode()
	return framer.NewJSONFramedReader(r)
}

// Framer is the default JSON framing behavior, with newlines delimiting individual objects.
var YAMLFramer = yamlFramer{}

type yamlFramer struct{}

// NewFrameWriter implements stream framing for this serializer
func (yamlFramer) NewFrameWriter(w io.Writer) io.Writer {
	return yamlFrameWriter{w}
}

// NewFrameReader implements stream framing for this serializer
func (yamlFramer) NewFrameReader(r io.ReadCloser) io.ReadCloser {
	// extract the YAML document chunks directly
	return utilyaml.NewDocumentDecoder(r)
}

type yamlFrameWriter struct {
	w io.Writer
}

// Write separates each document with the YAML document separator (`---` followed by line
// break). Writers must write well formed YAML documents (include a final line break).
func (w yamlFrameWriter) Write(data []byte) (n int, err error) {
	if _, err := w.w.Write([]byte("---\n")); err != nil {
		return 0, err
	}
	return w.w.Write(data)
}
