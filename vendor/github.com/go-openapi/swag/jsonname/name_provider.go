// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonname

import (
	"reflect"
	"strings"
	"sync"
)

// DefaultJSONNameProvider is the default cache for types.
var DefaultJSONNameProvider = NewNameProvider()

// NameProvider represents an object capable of translating from go property names
// to json property names.
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

// NewNameProvider creates a new name provider
func NewNameProvider() *NameProvider {
	return &NameProvider{
		lock:  &sync.Mutex{},
		index: make(map[reflect.Type]nameIndex),
	}
}

func buildnameIndex(tpe reflect.Type, idx, reverseIdx map[string]string) {
	for i := 0; i < tpe.NumField(); i++ {
		targetDes := tpe.Field(i)

		if targetDes.PkgPath != "" { // unexported
			continue
		}

		if targetDes.Anonymous { // walk embedded structures tree down first
			buildnameIndex(targetDes.Type, idx, reverseIdx)
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
	var idx = make(map[string]string, tpe.NumField())
	var reverseIdx = make(map[string]string, tpe.NumField())

	buildnameIndex(tpe, idx, reverseIdx)
	return nameIndex{jsonNames: idx, goNames: reverseIdx}
}

// GetJSONNames gets all the json property names for a type
func (n *NameProvider) GetJSONNames(subject any) []string {
	n.lock.Lock()
	defer n.lock.Unlock()
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
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

// GetJSONName gets the json name for a go property name
func (n *NameProvider) GetJSONName(subject any, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	return n.GetJSONNameForType(tpe, name)
}

// GetJSONNameForType gets the json name for a go property name on a given type
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

// GetGoName gets the go name for a json property name
func (n *NameProvider) GetGoName(subject any, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	return n.GetGoNameForType(tpe, name)
}

// GetGoNameForType gets the go name for a given type for a json property name
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
