/*
Copyright 2023 The Kubernetes Authors.

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

package lazy

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	"k8s.io/apiserver/pkg/cel"
)

type GetFieldFunc func(*MapValue) ref.Val

var _ ref.Val = (*MapValue)(nil)
var _ traits.Mapper = (*MapValue)(nil)

// MapValue is a map that lazily evaluate its value when a field is first accessed.
// The map value is not designed to be thread-safe.
type MapValue struct {
	typeValue *types.TypeValue

	// values are previously evaluated values obtained from callbacks
	values map[string]ref.Val
	// callbacks are a map of field name to the function that returns the field Val
	callbacks map[string]GetFieldFunc
	// knownValues are registered names, used for iteration
	knownValues []string
}

func NewMapValue(objectType ref.Type) *MapValue {
	return &MapValue{
		typeValue: types.NewTypeValue(objectType.TypeName(), traits.IndexerType|traits.FieldTesterType|traits.IterableType),
		values:    map[string]ref.Val{},
		callbacks: map[string]GetFieldFunc{},
	}
}

// Append adds the given field with its name and callback.
func (m *MapValue) Append(name string, callback GetFieldFunc) {
	m.knownValues = append(m.knownValues, name)
	m.callbacks[name] = callback
}

// Contains checks if the key is known to the map
func (m *MapValue) Contains(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if v != nil && types.IsUnknownOrError(v) {
		return v
	}
	return types.Bool(found)
}

// Iterator returns an iterator to traverse the map.
func (m *MapValue) Iterator() traits.Iterator {
	return &iterator{parent: m, index: 0}
}

// Size returns the number of currently known fields
func (m *MapValue) Size() ref.Val {
	return types.Int(len(m.callbacks))
}

// ConvertToNative returns an error because it is disallowed
func (m *MapValue) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("disallowed conversion from %q to %q", m.typeValue.TypeName(), typeDesc.Name())
}

// ConvertToType converts the map to the given type.
// Only its own type and "Type" type are allowed.
func (m *MapValue) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case m.typeValue:
		return m
	case types.TypeType:
		return m.typeValue
	}
	return types.NewErr("disallowed conversion from %q to %q", m.typeValue.TypeName(), typeVal.TypeName())
}

// Equal returns true if the other object is the same pointer-wise.
func (m *MapValue) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(*MapValue)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(m == otherMap)
}

// Type returns its registered type.
func (m *MapValue) Type() ref.Type {
	return m.typeValue
}

// Value is not allowed.
func (m *MapValue) Value() any {
	return types.NoSuchOverloadErr()
}

// resolveField resolves the field. Calls the callback if the value is not yet stored.
func (m *MapValue) resolveField(name string) ref.Val {
	v, seen := m.values[name]
	if seen {
		return v
	}
	f := m.callbacks[name]
	v = f(m)
	m.values[name] = v
	return v
}

func (m *MapValue) Find(key ref.Val) (ref.Val, bool) {
	n, ok := key.(types.String)
	if !ok {
		return types.MaybeNoSuchOverloadErr(n), true
	}
	name, ok := cel.Unescape(n.Value().(string))
	if !ok {
		return nil, false
	}
	if _, exists := m.callbacks[name]; !exists {
		return nil, false
	}
	return m.resolveField(name), true
}

func (m *MapValue) Get(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if found {
		return v
	}
	return types.ValOrErr(key, "no such key: %v", key)
}

type iterator struct {
	parent *MapValue
	index  int
}

func (i *iterator) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("disallowed conversion to %q", typeDesc.Name())
}

func (i *iterator) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("disallowed conversion o %q", typeValue.TypeName())
}

func (i *iterator) Equal(other ref.Val) ref.Val {
	otherIterator, ok := other.(*iterator)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(otherIterator == i)
}

func (i *iterator) Type() ref.Type {
	return types.IteratorType
}

func (i *iterator) Value() any {
	return nil
}

func (i *iterator) HasNext() ref.Val {
	return types.Bool(i.index < len(i.parent.knownValues))
}

func (i *iterator) Next() ref.Val {
	ret := i.parent.Get(types.String(i.parent.knownValues[i.index]))
	i.index++
	return ret
}

var _ traits.Iterator = (*iterator)(nil)
