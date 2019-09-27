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
	"strconv"
	"unsafe"

	jsoniter "github.com/json-iterator/go"
	"github.com/modern-go/reflect2"
	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
	"k8s.io/apimachinery/pkg/util/framer"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
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

type customNumberExtension struct {
	jsoniter.DummyExtension
}

func (cne *customNumberExtension) CreateDecoder(typ reflect2.Type) jsoniter.ValDecoder {
	if typ.String() == "interface {}" {
		return customNumberDecoder{}
	}
	return nil
}

type customNumberDecoder struct {
}

func (customNumberDecoder) Decode(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
	switch iter.WhatIsNext() {
	case jsoniter.NumberValue:
		var number jsoniter.Number
		iter.ReadVal(&number)
		i64, err := strconv.ParseInt(string(number), 10, 64)
		if err == nil {
			*(*interface{})(ptr) = i64
			return
		}
		f64, err := strconv.ParseFloat(string(number), 64)
		if err == nil {
			*(*interface{})(ptr) = f64
			return
		}
		iter.ReportError("DecodeNumber", err.Error())
	default:
		// init depth, if needed
		if iter.Attachment == nil {
			iter.Attachment = int(1)
		}

		// remember current depth
		originalAttachment := iter.Attachment

		// increment depth before descending
		if i, ok := iter.Attachment.(int); ok {
			iter.Attachment = i + 1
			if i > 10000 {
				iter.ReportError("parse", "exceeded max depth")
				return
			}
		}

		*(*interface{})(ptr) = iter.Read()

		// restore current depth
		iter.Attachment = originalAttachment
	}
}

// CaseSensitiveJsonIterator returns a jsoniterator API that's configured to be
// case-sensitive when unmarshalling, and otherwise compatible with
// the encoding/json standard library.
func CaseSensitiveJsonIterator() jsoniter.API {
	config := jsoniter.Config{
		EscapeHTML:             true,
		SortMapKeys:            true,
		ValidateJsonRawMessage: true,
		CaseSensitive:          true,
	}.Froze()
	// Force jsoniter to decode number to interface{} via int64/float64, if possible.
	config.RegisterExtension(&customNumberExtension{})
	return config
}

// Private copy of jsoniter to try to shield against possible mutations
// from outside. Still does not protect from package level jsoniter.Register*() functions - someone calling them
// in some other library will mess with every usage of the jsoniter library in the whole program.
// See https://github.com/json-iterator/go/issues/265
var caseSensitiveJsonIterator = CaseSensitiveJsonIterator()

// gvkWithDefaults returns group kind and version defaulting from provided default
func gvkWithDefaults(actual, defaultGVK schema.GroupVersionKind) schema.GroupVersionKind {
	if len(actual.Kind) == 0 {
		actual.Kind = defaultGVK.Kind
	}
	if len(actual.Version) == 0 && len(actual.Group) == 0 {
		actual.Group = defaultGVK.Group
		actual.Version = defaultGVK.Version
	}
	if len(actual.Version) == 0 && actual.Group == defaultGVK.Group {
		actual.Version = defaultGVK.Version
	}
	return actual
}

// Decode attempts to convert the provided data into YAML or JSON, extract the stored schema kind, apply the provided default gvk, and then
// load that data into an object matching the desired schema kind or the provided into.
// If into is *runtime.Unknown, the raw data will be extracted and no decoding will be performed.
// If into is not registered with the typer, then the object will be straight decoded using normal JSON/YAML unmarshalling.
// If into is provided and the original data is not fully qualified with kind/version/group, the type of the into will be used to alter the returned gvk.
// If into is nil or data's gvk different from into's gvk, it will generate a new Object with ObjectCreater.New(gvk)
// On success or most errors, the method will return the calculated schema kind.
// The gvk calculate priority will be originalData > default gvk > into
func (s *Serializer) Decode(originalData []byte, gvk *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
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
		*actual = gvkWithDefaults(*actual, *gvk)
	}

	if unk, ok := into.(*runtime.Unknown); ok && unk != nil {
		unk.Raw = originalData
		unk.ContentType = runtime.ContentTypeJSON
		unk.GetObjectKind().SetGroupVersionKind(*actual)
		return unk, actual, nil
	}

	if into != nil {
		_, isUnstructured := into.(runtime.Unstructured)
		types, _, err := s.typer.ObjectKinds(into)
		switch {
		case runtime.IsNotRegisteredError(err), isUnstructured:
			if err := caseSensitiveJsonIterator.Unmarshal(data, into); err != nil {
				return nil, actual, err
			}
			return into, actual, nil
		case err != nil:
			return nil, actual, err
		default:
			*actual = gvkWithDefaults(*actual, types[0])
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

	if err := caseSensitiveJsonIterator.Unmarshal(data, obj); err != nil {
		return nil, actual, err
	}
	return obj, actual, nil
}

// Encode serializes the provided object to the given writer.
func (s *Serializer) Encode(obj runtime.Object, w io.Writer) error {
	if s.yaml {
		json, err := caseSensitiveJsonIterator.Marshal(obj)
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
		data, err := caseSensitiveJsonIterator.MarshalIndent(obj, "", "  ")
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
	_, _, ok = utilyaml.GuessJSONStream(peek, 2048)
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

// YAMLFramer is the default JSON framing behavior, with newlines delimiting individual objects.
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
