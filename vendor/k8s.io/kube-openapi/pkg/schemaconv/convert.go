/*
Copyright 2022 The Kubernetes Authors.

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

package schemaconv

import (
	"fmt"

	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
)

// Basic conversion functions to convert OpenAPI schema definitions to
// SMD Schema atoms
func convertPrimitive(typ string, format string) (a schema.Atom) {
	switch typ {
	case proto.Integer:
		a.Scalar = ptr(schema.Numeric)
	case proto.Number:
		a.Scalar = ptr(schema.Numeric)
	case proto.String:
		switch format {
		case "":
			a.Scalar = ptr(schema.String)
		case "byte":
			// byte really means []byte and is encoded as a string.
			a.Scalar = ptr(schema.String)
		case "int-or-string":
			a.Scalar = ptr(schema.Scalar("untyped"))
		case "date-time":
			a.Scalar = ptr(schema.Scalar("untyped"))
		default:
			a.Scalar = ptr(schema.Scalar("untyped"))
		}
	case proto.Boolean:
		a.Scalar = ptr(schema.Boolean)
	default:
		a.Scalar = ptr(schema.Scalar("untyped"))
	}

	return a
}

func convertArray(ext map[string]any, elementType schema.TypeRef) (atom schema.Atom, err error) {
	atom.List = &schema.List{
		ElementRelationship: schema.Atomic,
	}
	l := atom.List
	l.ElementType = elementType

	if val, ok := ext["x-kubernetes-list-type"]; ok {
		switch val {
		case "atomic":
			l.ElementRelationship = schema.Atomic
		case "set":
			l.ElementRelationship = schema.Associative
		case "map":
			l.ElementRelationship = schema.Associative
			if keys, ok := ext["x-kubernetes-list-map-keys"]; ok {
				if keyNames, ok := toStringSlice(keys); ok {
					l.Keys = keyNames
				} else {
					err = fmt.Errorf("uninterpreted map keys: %#v", keys)
				}
			} else {
				err = fmt.Errorf("missing map keys")
			}
		default:
			err = fmt.Errorf("unknown list type: %v", val)
		}
	} else if val, ok := ext["x-kubernetes-patch-strategy"]; ok {
		switch val {
		case "merge", "merge,retainKeys":
			l.ElementRelationship = schema.Associative
			if key, ok := ext["x-kubernetes-patch-merge-key"]; ok {
				if keyName, ok := key.(string); ok {
					l.Keys = []string{keyName}
				} else {
					err = fmt.Errorf("uninterpreted merge key: %#v", key)
				}
			} else {
				// It's not an error for this to be absent, it
				// means it's a set.
			}
		case "retainKeys":
			break
		default:
			err = fmt.Errorf("unknown patch strategy %v", val)
		}
	}

	return atom, err
}

func toStringSlice(o interface{}) (out []string, ok bool) {
	switch t := o.(type) {
	case []interface{}:
		for _, v := range t {
			switch vt := v.(type) {
			case string:
				out = append(out, vt)
			}
		}
		return out, true
	case []string:
		return t, true
	}
	return nil, false
}
