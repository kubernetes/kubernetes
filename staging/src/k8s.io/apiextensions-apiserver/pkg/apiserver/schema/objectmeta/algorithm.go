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

package objectmeta

import (
	"sort"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// CoerceOptions gives the ability to ReturnUnknownFieldPaths for fields
// unrecognized by the schema or DropInvalidFields for fields that are a part
// of the schema, but are malformed.
type CoerceOptions struct {
	// DropInvalidFields discards malformed serialized metadata fields that
	// cannot be successfully decoded to the corresponding ObjectMeta field.
	// This only applies to fields that are recognized as part of the schema,
	// but of an invalid type (i.e. cause an error when unmarshaling, rather
	// than being dropped or causing a strictErr).
	DropInvalidFields bool
	// ReturnUnknownFieldPaths will return the paths to fields that are not
	// recognized as part of the schema.
	ReturnUnknownFieldPaths bool
}

// Coerce checks types of embedded ObjectMeta and TypeMeta and prunes unknown fields inside the former.
// It does coerce ObjectMeta and TypeMeta at the root if isResourceRoot is true.
// If opts.ReturnUnknownFieldPaths is true, it will return the paths of any fields that are not a part of the
// schema that are dropped when unmarshaling.
// If opts.DropInvalidFields is true, fields of wrong type will be dropped.
func CoerceWithOptions(pth *field.Path, obj interface{}, s *structuralschema.Structural, isResourceRoot bool, opts CoerceOptions) (*field.Error, []string) {
	if isResourceRoot {
		if s == nil {
			s = &structuralschema.Structural{}
		}
		if !s.XEmbeddedResource {
			clone := *s
			clone.XEmbeddedResource = true
			s = &clone
		}
	}
	c := coercer{DropInvalidFields: opts.DropInvalidFields, ReturnUnknownFieldPaths: opts.ReturnUnknownFieldPaths}
	schemaOpts := &structuralschema.UnknownFieldPathOptions{
		TrackUnknownFieldPaths: opts.ReturnUnknownFieldPaths,
	}
	fieldErr := c.coerce(pth, obj, s, schemaOpts)
	sort.Strings(schemaOpts.UnknownFieldPaths)
	return fieldErr, schemaOpts.UnknownFieldPaths
}

// Coerce calls CoerceWithOptions without returning unknown field paths.
func Coerce(pth *field.Path, obj interface{}, s *structuralschema.Structural, isResourceRoot, dropInvalidFields bool) *field.Error {
	fieldErr, _ := CoerceWithOptions(pth, obj, s, isResourceRoot, CoerceOptions{DropInvalidFields: dropInvalidFields})
	return fieldErr
}

type coercer struct {
	DropInvalidFields       bool
	ReturnUnknownFieldPaths bool
}

func (c *coercer) coerce(pth *field.Path, x interface{}, s *structuralschema.Structural, opts *structuralschema.UnknownFieldPathOptions) *field.Error {
	if s == nil {
		return nil
	}
	origPathLen := len(opts.ParentPath)
	defer func() {
		opts.ParentPath = opts.ParentPath[:origPathLen]
	}()
	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			if s.XEmbeddedResource {
				switch k {
				case "apiVersion", "kind":
					if _, ok := v.(string); !ok && c.DropInvalidFields {
						delete(x, k)
					} else if !ok {
						return field.Invalid(pth.Child(k), v, "must be a string")
					}
				case "metadata":
					meta, found, unknownFields, err := GetObjectMetaWithOptions(x, ObjectMetaOptions{
						DropMalformedFields:     c.DropInvalidFields,
						ReturnUnknownFieldPaths: c.ReturnUnknownFieldPaths,
						ParentPath:              pth,
					})
					opts.UnknownFieldPaths = append(opts.UnknownFieldPaths, unknownFields...)
					if err != nil {
						if !c.DropInvalidFields {
							return field.Invalid(pth.Child("metadata"), v, err.Error())
						}
						// pass through on error if DropInvalidFields is true
					} else if found {
						if err := SetObjectMeta(x, meta); err != nil {
							return field.Invalid(pth.Child("metadata"), v, err.Error())
						}
						if meta.CreationTimestamp.IsZero() {
							unstructured.RemoveNestedField(x, "metadata", "creationTimestamp")
						}
					}
				}
			}
			prop, ok := s.Properties[k]
			if ok {
				opts.AppendKey(k)
				if err := c.coerce(pth.Child(k), v, &prop, opts); err != nil {
					return err
				}
				opts.ParentPath = opts.ParentPath[:origPathLen]
			} else if s.AdditionalProperties != nil {
				opts.AppendKey(k)
				if err := c.coerce(pth.Key(k), v, s.AdditionalProperties.Structural, opts); err != nil {
					return err
				}
				opts.ParentPath = opts.ParentPath[:origPathLen]
			}
		}
	case []interface{}:
		for i, v := range x {
			opts.AppendIndex(i)
			if err := c.coerce(pth.Index(i), v, s.Items, opts); err != nil {
				return err
			}
			opts.ParentPath = opts.ParentPath[:origPathLen]
		}
	default:
		// scalars, do nothing
	}

	return nil
}
