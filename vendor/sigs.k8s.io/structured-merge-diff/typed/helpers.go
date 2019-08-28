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
	Path         fieldpath.Path
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

// errorFormatter makes it easy to keep a list of validation errors. They
// should all be packed into a single error object before leaving the package
// boundary, since it's weird to have functions not return a plain error type.
type errorFormatter struct {
	path fieldpath.Path
}

func (ef *errorFormatter) descend(pe fieldpath.PathElement) {
	ef.path = append(ef.path, pe)
}

// parent returns the parent, for the purpose of buffer reuse. It's an error to
// call this if there is no parent.
func (ef *errorFormatter) parent() errorFormatter {
	return errorFormatter{
		path: ef.path[:len(ef.path)-1],
	}
}

func (ef errorFormatter) errorf(format string, args ...interface{}) ValidationErrors {
	return ValidationErrors{{
		Path:         append(fieldpath.Path{}, ef.path...),
		ErrorMessage: fmt.Sprintf(format, args...),
	}}
}

func (ef errorFormatter) error(err error) ValidationErrors {
	return ValidationErrors{{
		Path:         append(fieldpath.Path{}, ef.path...),
		ErrorMessage: err.Error(),
	}}
}

func (ef errorFormatter) prefixError(prefix string, err error) ValidationErrors {
	return ValidationErrors{{
		Path:         append(fieldpath.Path{}, ef.path...),
		ErrorMessage: prefix + err.Error(),
	}}
}

type atomHandler interface {
	doScalar(schema.Scalar) ValidationErrors
	doList(schema.List) ValidationErrors
	doMap(schema.Map) ValidationErrors

	errorf(msg string, args ...interface{}) ValidationErrors
}

func resolveSchema(s *schema.Schema, tr schema.TypeRef, v *value.Value, ah atomHandler) ValidationErrors {
	a, ok := s.Resolve(tr)
	if !ok {
		return ah.errorf("schema error: no type found matching: %v", *tr.NamedType)
	}

	a = deduceAtom(a, v)
	return handleAtom(a, tr, ah)
}

func deduceAtom(a schema.Atom, v *value.Value) schema.Atom {
	switch {
	case v == nil:
	case v.FloatValue != nil, v.IntValue != nil, v.StringValue != nil, v.BooleanValue != nil:
		return schema.Atom{Scalar: a.Scalar}
	case v.ListValue != nil:
		return schema.Atom{List: a.List}
	case v.MapValue != nil:
		return schema.Atom{Map: a.Map}
	}
	return a
}

func handleAtom(a schema.Atom, tr schema.TypeRef, ah atomHandler) ValidationErrors {
	switch {
	case a.Map != nil:
		return ah.doMap(*a.Map)
	case a.Scalar != nil:
		return ah.doScalar(*a.Scalar)
	case a.List != nil:
		return ah.doList(*a.List)
	}

	name := "inlined"
	if tr.NamedType != nil {
		name = "named type: " + *tr.NamedType
	}

	return ah.errorf("schema error: invalid atom: %v", name)
}

func (ef errorFormatter) validateScalar(t schema.Scalar, v *value.Value, prefix string) (errs ValidationErrors) {
	if v == nil {
		return nil
	}
	if v.Null {
		return nil
	}
	switch t {
	case schema.Numeric:
		if v.FloatValue == nil && v.IntValue == nil {
			// TODO: should the schema separate int and float?
			return ef.errorf("%vexpected numeric (int or float), got %v", prefix, v)
		}
	case schema.String:
		if v.StringValue == nil {
			return ef.errorf("%vexpected string, got %v", prefix, v)
		}
	case schema.Boolean:
		if v.BooleanValue == nil {
			return ef.errorf("%vexpected boolean, got %v", prefix, v)
		}
	}
	return nil
}

// Returns the list, or an error. Reminder: nil is a valid list and might be returned.
func listValue(val value.Value) (*value.List, error) {
	switch {
	case val.Null:
		// Null is a valid list.
		return nil, nil
	case val.ListValue != nil:
		return val.ListValue, nil
	default:
		return nil, fmt.Errorf("expected list, got %v", val)
	}
}

// Returns the map, or an error. Reminder: nil is a valid map and might be returned.
func mapValue(val value.Value) (*value.Map, error) {
	switch {
	case val.Null:
		return nil, nil
	case val.MapValue != nil:
		return val.MapValue, nil
	default:
		return nil, fmt.Errorf("expected map, got %v", val)
	}
}

func keyedAssociativeListItemToPathElement(list schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	if child.Null {
		// For now, the keys are required which means that null entries
		// are illegal.
		return pe, errors.New("associative list with keys may not have a null element")
	}
	if child.MapValue == nil {
		return pe, errors.New("associative list with keys may not have non-map elements")
	}
	keyMap := &value.Map{}
	for _, fieldName := range list.Keys {
		var fieldValue value.Value
		field, ok := child.MapValue.Get(fieldName)
		if ok {
			fieldValue = field.Value
		} else {
			// Treat keys as required.
			return pe, fmt.Errorf("associative list with keys has an element that omits key field %q", fieldName)
		}
		keyMap.Set(fieldName, fieldValue)
	}
	pe.Key = keyMap
	return pe, nil
}

func setItemToPathElement(list schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	switch {
	case child.MapValue != nil:
		// TODO: atomic maps should be acceptable.
		return pe, errors.New("associative list without keys has an element that's a map type")
	case child.ListValue != nil:
		// Should we support a set of lists? For the moment
		// let's say we don't.
		// TODO: atomic lists should be acceptable.
		return pe, errors.New("not supported: associative list with lists as elements")
	case child.Null:
		return pe, errors.New("associative list without keys has an element that's an explicit null")
	default:
		// We are a set type.
		pe.Value = &child
		return pe, nil
	}
}

func listItemToPathElement(list schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
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
