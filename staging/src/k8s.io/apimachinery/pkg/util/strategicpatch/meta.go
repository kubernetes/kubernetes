/*
Copyright 2017 The Kubernetes Authors.

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

package strategicpatch

import (
	"reflect"

	"k8s.io/apimachinery/pkg/util/mergepatch"
	forkedjson "k8s.io/apimachinery/third_party/forked/golang/json"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

type PatchMeta struct {
	PatchStrategies []string
	PatchMergeKey   string
}

type LookupPatchMeta interface {
	// Lookup the patch metadata (patch strategy and merge key)
	LookupPatchMetadata(key string) (LookupPatchMeta, *PatchMeta, error)
	// Get the type name of the field
	Name() string
}

type PatchMetaFromOpenAPI struct {
	Schema openapi.Schema
}

func NewPatchMetaFromOpenAPI(s openapi.Schema) PatchMetaFromOpenAPI {
	return PatchMetaFromOpenAPI{Schema: s}
}

var _ LookupPatchMeta = PatchMetaFromOpenAPI{}

func (s PatchMetaFromOpenAPI) LookupPatchMetadata(key string) (LookupPatchMeta, *PatchMeta, error) {
	if s.Schema == nil {
		return nil, &PatchMeta{}, nil
	}
	patchItem := NewPatchItem(key, s.Schema.GetPath())
	s.Schema.Accept(&patchItem)

	err := patchItem.Error()
	if err != nil {
		return nil, nil, err
	}
	return PatchMetaFromOpenAPI{Schema: patchItem.subschema},
		&patchItem.patchmeta, nil
}

func (s PatchMetaFromOpenAPI) Name() string {
	return s.Schema.GetName()
}

type PatchMetaFromStruct struct {
	T reflect.Type
}

func NewPatchMetaFromStruct(dataStruct interface{}) (PatchMetaFromStruct, error) {
	t, err := getTagStructType(dataStruct)
	return PatchMetaFromStruct{T: t}, err
}

var _ LookupPatchMeta = PatchMetaFromStruct{}

func (s PatchMetaFromStruct) LookupPatchMetadata(key string) (LookupPatchMeta, *PatchMeta, error) {
	fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(s.T, key)
	if err != nil {
		return nil, nil, err
	}

	return PatchMetaFromStruct{T: fieldType},
		&PatchMeta{
			PatchStrategies: fieldPatchStrategies,
			PatchMergeKey:   fieldPatchMergeKey,
		}, nil
}

func (s PatchMetaFromStruct) Name() string {
	return s.T.Kind().String()
}

func getTagStructType(dataStruct interface{}) (reflect.Type, error) {
	if dataStruct == nil {
		return nil, mergepatch.ErrBadArgKind(struct{}{}, nil)
	}

	t := reflect.TypeOf(dataStruct)
	// Get the underlying type for pointers
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return nil, mergepatch.ErrBadArgKind(struct{}{}, dataStruct)
	}

	return t, nil
}

func GetTagStructTypeOrDie(dataStruct interface{}) reflect.Type {
	t, err := getTagStructType(dataStruct)
	if err != nil {
		panic(err)
	}
	return t
}
