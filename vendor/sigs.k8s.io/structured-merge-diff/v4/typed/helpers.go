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

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/value"
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

// WithPrefix prefixes all errors path with the given pathelement. This
// is useful when unwinding the stack on errors.
func (errs ValidationErrors) WithPrefix(prefix string) ValidationErrors {
	for i := range errs {
		errs[i].Path = prefix + errs[i].Path
	}
	return errs
}

// WithLazyPrefix prefixes all errors path with the given pathelement.
// This is useful when unwinding the stack on errors. Prefix is
// computed lazily only if there is an error.
func (errs ValidationErrors) WithLazyPrefix(fn func() string) ValidationErrors {
	if len(errs) == 0 {
		return errs
	}
	prefix := ""
	if fn != nil {
		prefix = fn()
	}
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

func resolveSchema(s *schema.Schema, tr schema.TypeRef, v value.Value, ah atomHandler) ValidationErrors {
	a, ok := s.Resolve(tr)
	if !ok {
		return errorf("schema error: no type found matching: %v", *tr.NamedType)
	}

	a = deduceAtom(a, v)
	return handleAtom(a, tr, ah)
}

// deduceAtom determines which of the possible types in atom 'atom' applies to value 'val'.
// If val is of a type allowed by atom, return a copy of atom with all other types set to nil.
// if val is nil, or is not of a type allowed by atom, just return the original atom,
// and validation will fail at a later stage. (with a more useful error)
func deduceAtom(atom schema.Atom, val value.Value) schema.Atom {
	switch {
	case val == nil:
	case val.IsFloat(), val.IsInt(), val.IsString(), val.IsBool():
		if atom.Scalar != nil {
			return schema.Atom{Scalar: atom.Scalar}
		}
	case val.IsList():
		if atom.List != nil {
			return schema.Atom{List: atom.List}
		}
	case val.IsMap():
		if atom.Map != nil {
			return schema.Atom{Map: atom.Map}
		}
	}
	return atom
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
func listValue(a value.Allocator, val value.Value) (value.List, error) {
	if val.IsNull() {
		// Null is a valid list.
		return nil, nil
	}
	if !val.IsList() {
		return nil, fmt.Errorf("expected list, got %v", val)
	}
	return val.AsListUsing(a), nil
}

// Returns the map, or an error. Reminder: nil is a valid map and might be returned.
func mapValue(a value.Allocator, val value.Value) (value.Map, error) {
	if val == nil {
		return nil, fmt.Errorf("expected map, got nil")
	}
	if val.IsNull() {
		// Null is a valid map.
		return nil, nil
	}
	if !val.IsMap() {
		return nil, fmt.Errorf("expected map, got %v", val)
	}
	return val.AsMapUsing(a), nil
}

func keyedAssociativeListItemToPathElement(a value.Allocator, list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	if child.IsNull() {
		// For now, the keys are required which means that null entries
		// are illegal.
		return pe, errors.New("associative list with keys may not have a null element")
	}
	if !child.IsMap() {
		return pe, errors.New("associative list with keys may not have non-map elements")
	}
	keyMap := value.FieldList{}
	m := child.AsMapUsing(a)
	defer a.Free(m)
	for _, fieldName := range list.Keys {
		if val, ok := m.Get(fieldName); ok {
			keyMap = append(keyMap, value.Field{Name: fieldName, Value: val})
		} else {
			return pe, fmt.Errorf("associative list with keys has an element that omits key field %q", fieldName)
		}
	}
	keyMap.Sort()
	pe.Key = &keyMap
	return pe, nil
}

func setItemToPathElement(list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	pe := fieldpath.PathElement{}
	switch {
	case child.IsMap():
		// TODO: atomic maps should be acceptable.
		return pe, errors.New("associative list without keys has an element that's a map type")
	case child.IsList():
		// Should we support a set of lists? For the moment
		// let's say we don't.
		// TODO: atomic lists should be acceptable.
		return pe, errors.New("not supported: associative list with lists as elements")
	case child.IsNull():
		return pe, errors.New("associative list without keys has an element that's an explicit null")
	default:
		// We are a set type.
		pe.Value = &child
		return pe, nil
	}
}

func listItemToPathElement(a value.Allocator, list *schema.List, index int, child value.Value) (fieldpath.PathElement, error) {
	if list.ElementRelationship == schema.Associative {
		if len(list.Keys) > 0 {
			return keyedAssociativeListItemToPathElement(a, list, index, child)
		}

		// If there's no keys, then we must be a set of primitives.
		return setItemToPathElement(list, index, child)
	}

	// Use the index as a key for atomic lists.
	return fieldpath.PathElement{Index: &index}, nil
}
