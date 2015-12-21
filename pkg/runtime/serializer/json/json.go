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

package json

import (
	"encoding/json"
	"io"

	"github.com/ghodss/yaml"
	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
)

// NewSerializer creates a JSON serializer that handles encoding versioned objects into the proper JSON form. If typer
// is not nil, the object has the group, version, and kind fields set.
func NewSerializer(meta MetaFactory, creater runtime.ObjectCreater, typer runtime.Typer, pretty bool) runtime.Serializer {
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
func NewYAMLSerializer(meta MetaFactory, creater runtime.ObjectCreater, typer runtime.Typer) runtime.Serializer {
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
	typer   runtime.Typer
	yaml    bool
	pretty  bool
}

func (s *Serializer) Decode(originalData []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
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

	if unk, ok := into.(*runtime.Unknown); ok && unk != nil {
		unk.GetObjectKind().SetGroupVersionKind(actual)
		return unk, actual, nil
	}

	if gvk != nil && (len(actual.Kind) == 0 || len(actual.Version) == 0) {
		// apply kind and version defaulting from provided default
		if len(actual.Kind) == 0 {
			actual.Kind = gvk.Kind
		}
		if len(actual.Version) == 0 && len(gvk.Version) > 0 {
			actual.Group = gvk.Group
			actual.Version = gvk.Version
		}
	}

	if into != nil {
		typed, _, err := s.typer.ObjectKind(into)
		switch {
		case runtime.IsNotRegisteredError(err):
			if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(into); err != nil {
				return nil, actual, err
			}
			return into, actual, nil
		case err != nil:
			return nil, actual, err
		default:
			if len(actual.Kind) == 0 {
				actual.Kind = typed.Kind
			}
			if len(actual.Version) == 0 {
				actual.Group = typed.Group
				actual.Version = typed.Version
			} else if len(actual.Group) == 0 {
				actual.Group = typed.Group
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

func (s *Serializer) EncodeToStream(obj runtime.Object, w io.Writer, overrides ...unversioned.GroupVersion) error {
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

func (s *Serializer) RecognizesData(peek io.Reader) (bool, error) {
	_, ok := utilyaml.GuessJSONStream(peek, 2048)
	if s.yaml {
		return !ok, nil
	}
	return ok, nil
}
