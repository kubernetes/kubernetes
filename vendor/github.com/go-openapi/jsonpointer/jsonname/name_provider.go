// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonname

import (
	"reflect"
	"strings"
	"sync"
)

// DefaultJSONNameProvider is the default cache for types.
var DefaultJSONNameProvider = NewNameProvider() //nolint:gochecknoglobals // default settings, for backward compatible package-level settings

var _ providerIface = (*NameProvider)(nil)

// NameProvider represents an object capable of translating from go property names to json property
// names.
//
// This type is thread-safe.
//
// See [github.com/go-openapi/jsonpointer.Pointer] for an example.
type NameProvider struct {
	lock  *sync.Mutex
	index map[reflect.Type]nameIndex
}

type nameIndex struct {
	jsonNames map[string]string
	goNames   map[string]string
}

// NewNameProvider creates a new name provider.
func NewNameProvider() *NameProvider {
	return &NameProvider{
		lock:  &sync.Mutex{},
		index: make(map[reflect.Type]nameIndex),
	}
}

func buildnameIndex(tpe reflect.Type, idx, reverseIdx map[string]string) {
	for targetDes := range tpe.Fields() {

		if targetDes.PkgPath != "" { // unexported
			continue
		}

		if targetDes.Anonymous { // walk embedded structures tree down first
			// An embedded field is not necessarily a struct: it may be a pointer to a struct, or a named
			// non-struct type (e.g. an embedded named slice or map). Only struct shapes carry promoted
			// fields worth indexing; anything else contributes no json name and must not be walked.
			if embedded := structTypeOf(targetDes.Type); embedded != nil {
				buildnameIndex(embedded, idx, reverseIdx)
			}

			continue
		}

		if tag := targetDes.Tag.Get("json"); tag != "" {

			parts := strings.Split(tag, ",")
			if len(parts) == 0 {
				continue
			}

			nm := parts[0]
			if nm == "-" {
				continue
			}
			if nm == "" { // empty string means we want to use the Go name
				nm = targetDes.Name
			}

			idx[nm] = targetDes.Name
			reverseIdx[targetDes.Name] = nm
		}
	}
}

func newNameIndex(tpe reflect.Type) nameIndex {
	tpe = structTypeOf(tpe)
	if tpe == nil {
		// only struct shapes carry json names: anything else indexes to nothing.
		return nameIndex{jsonNames: map[string]string{}, goNames: map[string]string{}}
	}

	idx := make(map[string]string, tpe.NumField())
	reverseIdx := make(map[string]string, tpe.NumField())

	buildnameIndex(tpe, idx, reverseIdx)
	return nameIndex{jsonNames: idx, goNames: reverseIdx}
}

// structTypeOf reduces tpe to the struct type it designates, dereferencing a pointer type.
//
// It returns nil when tpe is nil or does not designate a struct, so that callers may treat
// "nothing to index here" uniformly instead of panicking in [reflect.Type.NumField].
func structTypeOf(tpe reflect.Type) reflect.Type {
	if tpe == nil {
		return nil
	}

	if tpe.Kind() == reflect.Pointer {
		tpe = tpe.Elem()
	}

	if tpe.Kind() != reflect.Struct {
		return nil
	}

	return tpe
}

// typeOfSubject returns the type of a document subject, dereferencing pointers.
//
// It returns nil for an untyped nil subject, which resolves to an empty name index rather than
// panicking in [reflect.Value.Type].
func typeOfSubject(subject any) reflect.Type {
	rValue := reflect.Indirect(reflect.ValueOf(subject))
	if !rValue.IsValid() {
		return nil
	}

	return rValue.Type()
}

// GetJSONNames gets all the json property names for a type.
func (n *NameProvider) GetJSONNames(subject any) []string {
	n.lock.Lock()
	defer n.lock.Unlock()
	tpe := typeOfSubject(subject)
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}

	res := make([]string, 0, len(names.jsonNames))
	for k := range names.jsonNames {
		res = append(res, k)
	}
	return res
}

// GetJSONName gets the json name for a go property name.
func (n *NameProvider) GetJSONName(subject any, name string) (string, bool) {
	tpe := typeOfSubject(subject)
	return n.GetJSONNameForType(tpe, name)
}

// GetJSONNameForType gets the json name for a go property name on a given type.
func (n *NameProvider) GetJSONNameForType(tpe reflect.Type, name string) (string, bool) {
	n.lock.Lock()
	defer n.lock.Unlock()
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}
	nme, ok := names.goNames[name]
	return nme, ok
}

// GetGoName gets the go name for a json property name.
func (n *NameProvider) GetGoName(subject any, name string) (string, bool) {
	tpe := typeOfSubject(subject)
	return n.GetGoNameForType(tpe, name)
}

// GetGoNameForType gets the go name for a given type for a json property name.
func (n *NameProvider) GetGoNameForType(tpe reflect.Type, name string) (string, bool) {
	n.lock.Lock()
	defer n.lock.Unlock()
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}
	nme, ok := names.jsonNames[name]
	return nme, ok
}

func (n *NameProvider) makeNameIndex(tpe reflect.Type) nameIndex {
	names := newNameIndex(tpe)
	n.index[tpe] = names
	return names
}
