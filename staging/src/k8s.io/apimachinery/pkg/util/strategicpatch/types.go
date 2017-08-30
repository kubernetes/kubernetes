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
	"strings"

	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
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

type patchItem struct {
	key       string
	path      *openapi.Path
	err       error
	patchmeta PatchMeta
	subschema openapi.Schema
}

func NewPatchItem(key string, path *openapi.Path) patchItem {
	return patchItem{
		key:  key,
		path: path,
	}
}

var _ LookupPatchItem = &patchItem{}

func (item *patchItem) Error() error {
	return item.err
}

func (item *patchItem) Path() *openapi.Path {
	return item.path
}

func (item *patchItem) VisitPrimitive(schema *openapi.Primitive) {
	item.subschema = nil
	item.patchmeta = PatchMeta{
		PatchStrategies: []string{""},
		PatchMergeKey:   "",
	}
}

func (item *patchItem) VisitArray(schema *openapi.Array) {
	subschema := schema.SubType
	subschema.Accept(item)
}

func (item *patchItem) VisitMap(schema *openapi.Map) {
	item.subschema = schema.SubType
	item.patchmeta = PatchMeta{
		PatchStrategies: []string{""},
		PatchMergeKey:   "",
	}
}

func (item *patchItem) VisitReference(schema openapi.Reference) {
	// passthrough
	schema.SubSchema().Accept(item)
}

func (item *patchItem) VisitKind(schema *openapi.Kind) {
	subschema, ok := schema.Fields[item.key]
	if !ok {
		item.err = FieldNotFoundError{Path: schema.GetPath().String(), Field: item.key}
		return
	}

	mergeKey, patchStrategies, err := parsePatchMetadata(subschema.GetExtensions())
	if err != nil {
		item.err = err
		//return
	}
	item.patchmeta = PatchMeta{
		PatchStrategies: patchStrategies,
		PatchMergeKey:   mergeKey,
	}
	item.subschema = subschema
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
