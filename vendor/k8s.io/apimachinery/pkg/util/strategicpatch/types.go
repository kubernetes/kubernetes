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
	"strings"

	"k8s.io/apimachinery/pkg/util/mergepatch"
	openapi "k8s.io/kube-openapi/pkg/util/proto"
)

const (
	patchStrategyOpenapiextensionKey = "x-kubernetes-patch-strategy"
	patchMergeKeyOpenapiextensionKey = "x-kubernetes-patch-merge-key"
)

type LookupPatchItem interface {
	openapi.SchemaVisitor

	Error() error
	Path() *openapi.Path
}

type kindItem struct {
	key          string
	path         *openapi.Path
	err          error
	patchmeta    PatchMeta
	subschema    openapi.Schema
	hasVisitKind bool
}

func NewKindItem(key string, path *openapi.Path) *kindItem {
	return &kindItem{
		key:  key,
		path: path,
	}
}

var _ LookupPatchItem = &kindItem{}

func (item *kindItem) Error() error {
	return item.err
}

func (item *kindItem) Path() *openapi.Path {
	return item.path
}

func (item *kindItem) VisitPrimitive(schema *openapi.Primitive) {
	item.err = errors.New("expected kind, but got primitive")
}

func (item *kindItem) VisitArray(schema *openapi.Array) {
	item.err = errors.New("expected kind, but got slice")
}

func (item *kindItem) VisitMap(schema *openapi.Map) {
	item.err = errors.New("expected kind, but got map")
}

func (item *kindItem) VisitReference(schema openapi.Reference) {
	if !item.hasVisitKind {
		schema.SubSchema().Accept(item)
	}
}

func (item *kindItem) VisitKind(schema *openapi.Kind) {
	subschema, ok := schema.Fields[item.key]
	if !ok {
		item.err = FieldNotFoundError{Path: schema.GetPath().String(), Field: item.key}
		return
	}

	mergeKey, patchStrategies, err := parsePatchMetadata(subschema.GetExtensions())
	if err != nil {
		item.err = err
		return
	}
	item.patchmeta = PatchMeta{
		patchStrategies: patchStrategies,
		patchMergeKey:   mergeKey,
	}
	item.subschema = subschema
}

type sliceItem struct {
	key          string
	path         *openapi.Path
	err          error
	patchmeta    PatchMeta
	subschema    openapi.Schema
	hasVisitKind bool
}

func NewSliceItem(key string, path *openapi.Path) *sliceItem {
	return &sliceItem{
		key:  key,
		path: path,
	}
}

var _ LookupPatchItem = &sliceItem{}

func (item *sliceItem) Error() error {
	return item.err
}

func (item *sliceItem) Path() *openapi.Path {
	return item.path
}

func (item *sliceItem) VisitPrimitive(schema *openapi.Primitive) {
	item.err = errors.New("expected slice, but got primitive")
}

func (item *sliceItem) VisitArray(schema *openapi.Array) {
	if !item.hasVisitKind {
		item.err = errors.New("expected visit kind first, then visit array")
	}
	subschema := schema.SubType
	item.subschema = subschema
}

func (item *sliceItem) VisitMap(schema *openapi.Map) {
	item.err = errors.New("expected slice, but got map")
}

func (item *sliceItem) VisitReference(schema openapi.Reference) {
	if !item.hasVisitKind {
		schema.SubSchema().Accept(item)
	} else {
		item.subschema = schema.SubSchema()
	}
}

func (item *sliceItem) VisitKind(schema *openapi.Kind) {
	subschema, ok := schema.Fields[item.key]
	if !ok {
		item.err = FieldNotFoundError{Path: schema.GetPath().String(), Field: item.key}
		return
	}

	mergeKey, patchStrategies, err := parsePatchMetadata(subschema.GetExtensions())
	if err != nil {
		item.err = err
		return
	}
	item.patchmeta = PatchMeta{
		patchStrategies: patchStrategies,
		patchMergeKey:   mergeKey,
	}
	item.hasVisitKind = true
	subschema.Accept(item)
}

func parsePatchMetadata(extensions map[string]interface{}) (string, []string, error) {
	ps, foundPS := extensions[patchStrategyOpenapiextensionKey]
	var patchStrategies []string
	var mergeKey, patchStrategy string
	var ok bool
	if foundPS {
		patchStrategy, ok = ps.(string)
		if ok {
			patchStrategies = strings.Split(patchStrategy, ",")
		} else {
			return "", nil, mergepatch.ErrBadArgType(patchStrategy, ps)
		}
	}
	mk, foundMK := extensions[patchMergeKeyOpenapiextensionKey]
	if foundMK {
		mergeKey, ok = mk.(string)
		if !ok {
			return "", nil, mergepatch.ErrBadArgType(mergeKey, mk)
		}
	}
	return mergeKey, patchStrategies, nil
}
