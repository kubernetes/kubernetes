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

	version, kind, err := s.meta.Interpret(data)
	if err != nil {
		return nil, nil, err
	}

	actual := &unversioned.GroupVersionKind{Kind: kind}
	if gv, err := unversioned.ParseGroupVersion(version); err == nil {
		actual.Group = gv.Group
		actual.Version = gv.Version
	}

	// apply kind and version defaulting
	if len(actual.Kind) == 0 || len(actual.Version) == 0 {
		if gvk == nil {
			if into != nil {
				// try to default the missing value from into
				if typed, _, err := s.typer.ObjectVersionAndKind(into); err == nil {
					if len(actual.Kind) == 0 {
						actual.Kind = typed.Kind
					}
					if len(actual.Version) == 0 {
						actual.Version = typed.Version
					}
				}
				// if we are still incomplete, and into is available, decode it directly
				if len(actual.Kind) == 0 || len(actual.Version) == 0 {
					if err := json.Unmarshal(data, into); err != nil {
						return nil, actual, err
					}
					return into, actual, nil
				}
			}
		} else {
			if len(actual.Kind) == 0 {
				actual.Kind = gvk.Kind
			}
			if len(actual.Version) == 0 && len(gvk.Version) > 0 {
				actual.Group = gvk.Group
				actual.Version = gvk.Version
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
	if s.typer != nil {
		gvk, isUnversioned, err := s.typer.ObjectVersionAndKind(obj)
		if err != nil {
			return err
		}

		// apply appropriate overrides for the group version being targeted
		var gv unversioned.GroupVersion
		if isUnversioned {
			// unversioned types are targeted to the caller's first preferred group version
			if len(overrides) > 0 {
				gv = overrides[0]
			} else {
				gv = gvk.GroupVersion()
			}
		} else {
			if len(overrides) == 1 {
				// if only a single override is present, use that as the target version
				gv = overrides[0]
			} else {
				// if multiple or none are present, match to the preferred version
				gv = gvk.GroupVersion()
				for _, override := range overrides {
					if override.Group == gvk.Group {
						gv = override
						break
					}
				}
			}
		}
		if err := s.meta.Update(gv.String(), gvk.Kind, obj); err != nil {
			return err
		}
	}

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
