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
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Coerce checks types of embedded ObjectMeta and TypeMeta and prunes unknown fields inside the former.
// It does coerce ObjectMeta and TypeMeta at the root if isResourceRoot is true.
// If dropInvalidFields is true, fields of wrong type will be dropped.
func Coerce(pth *field.Path, obj interface{}, s *structuralschema.Structural, isResourceRoot, dropInvalidFields bool) *field.Error {
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
	c := coercer{dropInvalidFields: dropInvalidFields}
	return c.coerce(pth, obj, s)
}

type coercer struct {
	dropInvalidFields bool
}

func (c *coercer) coerce(pth *field.Path, x interface{}, s *structuralschema.Structural) *field.Error {
	if s == nil {
		return nil
	}
	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			if s.XEmbeddedResource {
				switch k {
				case "apiVersion", "kind":
					if _, ok := v.(string); !ok && c.dropInvalidFields {
						delete(x, k)
					} else if !ok {
						return field.Invalid(pth.Child(k), v, "must be a string")
					}
				case "metadata":
					meta, found, err := GetObjectMeta(x, c.dropInvalidFields)
					if err != nil {
						if !c.dropInvalidFields {
							return field.Invalid(pth.Child("metadata"), v, err.Error())
						}
						// pass through on error if dropInvalidFields is true
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
				if err := c.coerce(pth.Child(k), v, &prop); err != nil {
					return err
				}
			} else if s.AdditionalProperties != nil {
				if err := c.coerce(pth.Key(k), v, s.AdditionalProperties.Structural); err != nil {
					return err
				}
			}
		}
	case []interface{}:
		for i, v := range x {
			if err := c.coerce(pth.Index(i), v, s.Items); err != nil {
				return err
			}
		}
	default:
		// scalars, do nothing
	}

	return nil
}
