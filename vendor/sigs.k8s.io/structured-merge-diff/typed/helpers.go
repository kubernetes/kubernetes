/*
Copyright 2018 The Kubernetes Authors.

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

package typed

import (
	"errors"
	"fmt"
	"strings"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/schema"
	"sigs.k8s.io/structured-merge-diff/value"
)

// ValidationError reports an error about a particular field
type ValidationError struct {
	Path         string
	ErrorMessage string
}

// Error returns a human readable error message.
func (ve ValidationError) Error() string {
	if len(ve.Path) == 0 {
		return ve.ErrorMessage
	}
	return fmt.Sprintf("%s: %v", ve.Path, ve.ErrorMessage)
}

// ValidationErrors accumulates multiple validation error messages.
type ValidationErrors []ValidationError

// Error returns a human readable error message reporting each error in the
// list.
func (errs ValidationErrors) Error() string {
	if len(errs) == 1 {
		return errs[0].Error()
	}
	messages := []string{"errors:"}
	for _, e := range errs {
		messages = append(messages, "  "+e.Error())
	}
	return strings.Join(messages, "\n")
}

// Set the given path to all the validation errors.
func (errs ValidationErrors) WithPath(p string) ValidationErrors {
	for i := range errs {
		errs[i].Path = p
	}
	return errs
}

// Prefix all errors path with the given pathelement. This is useful
// when unwinding the stack on errors.
func (errs ValidationErrors) WithPrefix(prefix string) ValidationErrors {
	for i := range errs {
		errs[i].Path = prefix + errs[i].Path
	}
	return errs
}

func errorf(format string, args ...interface{}) ValidationErrors {
	return ValidationErrors{{
		ErrorMessage: fmt.Sprintf(format, args...),
	}}
}

type atomHandler interface {
	doScalar(*schema.Scalar) ValidationErrors
	doList(*schema.List) ValidationErrors
	doMap(*schema.Map) ValidationErrors
}

func resolveSchema(s *schema.Schema, tr schema.TypeRef, v *value.Value, ah atomHandler) ValidationErrors {
	a, ok := s.Resolve(tr)
	if !ok {
		return errorf("schema error: no type found matching: %v", *tr.NamedType)
	}

	a = deduceAtom(a, v)
	return handleAtom(a, tr, ah)
}

func deduceAtom(a schema.Atom, v *value.Value) schema.Atom {
	switch {
	case v == nil:
	case value.IsFloat(*v), value.IsInt(*v), value.IsString(*v), value.IsBool(*v):
		return schema.Atom{Scalar: a.Scalar}
	case value.IsList(*v):
		return schema.Atom{List: a.List}
	case value.IsMap(*v):
		return schema.Atom{Map: a.Map}
	}
	return a
}

func handleAtom(a schema.Atom, tr schema.TypeRef, ah atomHandler) ValidationErrors {
	switch {
	case a.Map != nil:
		return ah.doMap(a.Map)
	case a.Scalar != nil:
		return ah.doScalar(a.Scalar)
	case a.List != nil:
		return ah.doList(a.List)
	}

	name := "inlined"
	if tr.NamedType != nil {
		name = "named type: " + *tr.NamedType
	}

	return errorf("schema error: invalid atom: %v", name)
}

// Returns the list, or an error. Reminder: nil is a valid list and might be returned.
func listValue(val value.Value) ([]interface{}, error) {
	if val == nil {
		// Null is a valid list.
		return nil, nil
	}
	l, ok := val.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected list, got %v", val)
	}
	return l, nil
}

// Returns the map, or an error. Reminder: nil is a valid map and might be returned.
func mapValue(val value.Value) (value.Map, error) {
	if val == nil {
		// Null is a valid map.
		return nil, nil
	}
	if !value.IsMap(val) {
		return nil, fmt.Errorf("expected map, got %v", val)
	}
	return value.ValueMap(val), nil
}

func keyedAssociativeListItemToPathElement(list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	if child == nil {
		// For now, the keys are required which means that null entries
		// are illegal.
		return pe, errors.New("associative list with keys may not have a null element")
	}
	if !value.IsMap(child) {
		return pe, errors.New("associative list with keys may not have non-map elements")
	}
	m := value.ValueMap(child)
	keyValues := make([]fieldpath.KeyValue, 0, len(list.Keys))
	for _, fieldName := range list.Keys {
		if val, ok := m.Get(fieldName); ok {
			keyValues = append(keyValues, fieldpath.KeyValue{fieldName, val})
		} else {
			return pe, fmt.Errorf("associative list with keys has an element that omits key field %q", fieldName)
		}
	}
	fieldpath.SortKeyValues(keyValues)
	pe.Key = &keyValues

	return pe, nil
}

func setItemToPathElement(list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	switch {
	case value.IsMap(child):
		// TODO: atomic maps should be acceptable.
		return pe, errors.New("associative list without keys has an element that's a map type")
	case value.IsList(child):
		// Should we support a set of lists? For the moment
		// let's say we don't.
		// TODO: atomic lists should be acceptable.
		return pe, errors.New("not supported: associative list with lists as elements")
	case child == nil:
		return pe, errors.New("associative list without keys has an element that's an explicit null")
	default:
		// We are a set type.
		pe.Value = &child
		return pe, nil
	}
}

func listItemToPathElement(list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	if list.ElementRelationship == schema.Associative {
		if len(list.Keys) > 0 {
			return keyedAssociativeListItemToPathElement(list, index, child)
		}

		// If there's no keys, then we must be a set of primitives.
		return setItemToPathElement(list, index, child)
	}

	// Use the index as a key for atomic lists.
	return fieldpath.PathElement{Index: &index}, nil
}
