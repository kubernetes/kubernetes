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
	"errors"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/util/mergepatch"
	forkedjson "k8s.io/apimachinery/third_party/forked/golang/json"
	openapi "k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const patchMergeKey = "x-kubernetes-patch-merge-key"
const patchStrategy = "x-kubernetes-patch-strategy"

type PatchMeta struct {
	patchStrategies []string
	patchMergeKey   string
}

func (pm *PatchMeta) GetPatchStrategies() []string {
	if pm.patchStrategies == nil {
		return []string{}
	}
	return pm.patchStrategies
}

func (pm *PatchMeta) SetPatchStrategies(ps []string) {
	pm.patchStrategies = ps
}

func (pm *PatchMeta) GetPatchMergeKey() string {
	return pm.patchMergeKey
}

func (pm *PatchMeta) SetPatchMergeKey(pmk string) {
	pm.patchMergeKey = pmk
}

type LookupPatchMeta interface {
	// LookupPatchMetadataForStruct gets subschema and the patch metadata (e.g. patch strategy and merge key) for map.
	LookupPatchMetadataForStruct(key string) (LookupPatchMeta, PatchMeta, error)
	// LookupPatchMetadataForSlice get subschema and the patch metadata for slice.
	LookupPatchMetadataForSlice(key string) (LookupPatchMeta, PatchMeta, error)
	// Get the type name of the field
	Name() string
}

type PatchMetaFromStruct struct {
	T reflect.Type
}

func NewPatchMetaFromStruct(dataStruct interface{}) (PatchMetaFromStruct, error) {
	t, err := getTagStructType(dataStruct)
	return PatchMetaFromStruct{T: t}, err
}

var _ LookupPatchMeta = PatchMetaFromStruct{}

func (s PatchMetaFromStruct) LookupPatchMetadataForStruct(key string) (LookupPatchMeta, PatchMeta, error) {
	fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadataForStruct(s.T, key)
	if err != nil {
		return nil, PatchMeta{}, err
	}

	return PatchMetaFromStruct{T: fieldType},
		PatchMeta{
			patchStrategies: fieldPatchStrategies,
			patchMergeKey:   fieldPatchMergeKey,
		}, nil
}

func (s PatchMetaFromStruct) LookupPatchMetadataForSlice(key string) (LookupPatchMeta, PatchMeta, error) {
	subschema, patchMeta, err := s.LookupPatchMetadataForStruct(key)
	if err != nil {
		return nil, PatchMeta{}, err
	}
	elemPatchMetaFromStruct := subschema.(PatchMetaFromStruct)
	t := elemPatchMetaFromStruct.T

	var elemType reflect.Type
	switch t.Kind() {
	// If t is an array or a slice, get the element type.
	// If element is still an array or a slice, return an error.
	// Otherwise, return element type.
	case reflect.Array, reflect.Slice:
		elemType = t.Elem()
		if elemType.Kind() == reflect.Array || elemType.Kind() == reflect.Slice {
			return nil, PatchMeta{}, errors.New("unexpected slice of slice")
		}
	// If t is an pointer, get the underlying element.
	// If the underlying element is neither an array nor a slice, the pointer is pointing to a slice,
	// e.g. https://github.com/kubernetes/kubernetes/blob/bc22e206c79282487ea0bf5696d5ccec7e839a76/staging/src/k8s.io/apimachinery/pkg/util/strategicpatch/patch_test.go#L2782-L2822
	// If the underlying element is either an array or a slice, return its element type.
	case reflect.Pointer:
		t = t.Elem()
		if t.Kind() == reflect.Array || t.Kind() == reflect.Slice {
			t = t.Elem()
		}
		elemType = t
	default:
		return nil, PatchMeta{}, fmt.Errorf("expected slice or array type, but got: %s", s.T.Kind().String())
	}

	return PatchMetaFromStruct{T: elemType}, patchMeta, nil
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
	if t.Kind() == reflect.Pointer {
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

type PatchMetaFromOpenAPIV3 struct {
	// SchemaList is required to resolve OpenAPI V3 references
	SchemaList map[string]*spec.Schema
	Schema     *spec.Schema
}

func (s PatchMetaFromOpenAPIV3) traverse(key string) (PatchMetaFromOpenAPIV3, error) {
	if s.Schema == nil {
		return PatchMetaFromOpenAPIV3{}, nil
	}
	if len(s.Schema.Properties) == 0 {
		return PatchMetaFromOpenAPIV3{}, fmt.Errorf("unable to find api field \"%s\"", key)
	}
	subschema, ok := s.Schema.Properties[key]
	if !ok {
		return PatchMetaFromOpenAPIV3{}, fmt.Errorf("unable to find api field \"%s\"", key)
	}
	return PatchMetaFromOpenAPIV3{SchemaList: s.SchemaList, Schema: &subschema}, nil
}

func resolve(l *PatchMetaFromOpenAPIV3) error {
	if len(l.Schema.AllOf) > 0 {
		l.Schema = &l.Schema.AllOf[0]
	}
	if refString := l.Schema.Ref.String(); refString != "" {
		str := strings.TrimPrefix(refString, "#/components/schemas/")
		sch, ok := l.SchemaList[str]
		if ok {
			l.Schema = sch
		} else {
			return fmt.Errorf("unable to resolve %s in OpenAPI V3", refString)
		}
	}
	return nil
}

func (s PatchMetaFromOpenAPIV3) LookupPatchMetadataForStruct(key string) (LookupPatchMeta, PatchMeta, error) {
	l, err := s.traverse(key)
	if err != nil {
		return l, PatchMeta{}, err
	}
	p := PatchMeta{}
	f, ok := l.Schema.Extensions[patchMergeKey]
	if ok {
		p.SetPatchMergeKey(f.(string))
	}
	g, ok := l.Schema.Extensions[patchStrategy]
	if ok {
		p.SetPatchStrategies(strings.Split(g.(string), ","))
	}

	err = resolve(&l)
	return l, p, err
}

func (s PatchMetaFromOpenAPIV3) LookupPatchMetadataForSlice(key string) (LookupPatchMeta, PatchMeta, error) {
	l, err := s.traverse(key)
	if err != nil {
		return l, PatchMeta{}, err
	}
	p := PatchMeta{}
	f, ok := l.Schema.Extensions[patchMergeKey]
	if ok {
		p.SetPatchMergeKey(f.(string))
	}
	g, ok := l.Schema.Extensions[patchStrategy]
	if ok {
		p.SetPatchStrategies(strings.Split(g.(string), ","))
	}
	if l.Schema.Items != nil {
		l.Schema = l.Schema.Items.Schema
	}
	err = resolve(&l)
	return l, p, err
}

func (s PatchMetaFromOpenAPIV3) Name() string {
	schema := s.Schema
	if len(schema.Type) > 0 {
		return strings.Join(schema.Type, "")
	}
	return "Struct"
}

type PatchMetaFromOpenAPI struct {
	Schema openapi.Schema
}

func NewPatchMetaFromOpenAPI(s openapi.Schema) PatchMetaFromOpenAPI {
	return PatchMetaFromOpenAPI{Schema: s}
}

var _ LookupPatchMeta = PatchMetaFromOpenAPI{}

func (s PatchMetaFromOpenAPI) LookupPatchMetadataForStruct(key string) (LookupPatchMeta, PatchMeta, error) {
	if s.Schema == nil {
		return nil, PatchMeta{}, nil
	}
	kindItem := NewKindItem(key, s.Schema.GetPath())
	s.Schema.Accept(kindItem)

	err := kindItem.Error()
	if err != nil {
		return nil, PatchMeta{}, err
	}
	return PatchMetaFromOpenAPI{Schema: kindItem.subschema},
		kindItem.patchmeta, nil
}

func (s PatchMetaFromOpenAPI) LookupPatchMetadataForSlice(key string) (LookupPatchMeta, PatchMeta, error) {
	if s.Schema == nil {
		return nil, PatchMeta{}, nil
	}
	sliceItem := NewSliceItem(key, s.Schema.GetPath())
	s.Schema.Accept(sliceItem)

	err := sliceItem.Error()
	if err != nil {
		return nil, PatchMeta{}, err
	}
	return PatchMetaFromOpenAPI{Schema: sliceItem.subschema},
		sliceItem.patchmeta, nil
}

func (s PatchMetaFromOpenAPI) Name() string {
	schema := s.Schema
	return schema.GetName()
}
