// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"
	"strings"
	"sync"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/swag"
)

// ResolutionCache a cache for resolving urls
type ResolutionCache interface {
	Get(string) (interface{}, bool)
	Set(string, interface{})
}

type simpleCache struct {
	lock  sync.Mutex
	store map[string]interface{}
}

var resCache = initResolutionCache()

func initResolutionCache() ResolutionCache {
	return &simpleCache{store: map[string]interface{}{
		"http://swagger.io/v2/schema.json":       MustLoadSwagger20Schema(),
		"http://json-schema.org/draft-04/schema": MustLoadJSONSchemaDraft04(),
	}}
}

func (s *simpleCache) Get(uri string) (interface{}, bool) {
	s.lock.Lock()
	v, ok := s.store[uri]
	s.lock.Unlock()
	return v, ok
}

func (s *simpleCache) Set(uri string, data interface{}) {
	s.lock.Lock()
	s.store[uri] = data
	s.lock.Unlock()
}

// ResolveRef resolves a reference against a context root
func ResolveRef(root interface{}, ref *Ref) (*Schema, error) {
	resolver, err := defaultSchemaLoader(root, nil, nil)
	if err != nil {
		return nil, err
	}

	result := new(Schema)
	if err := resolver.Resolve(ref, result); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveParameter resolves a paramter reference against a context root
func ResolveParameter(root interface{}, ref Ref) (*Parameter, error) {
	resolver, err := defaultSchemaLoader(root, nil, nil)
	if err != nil {
		return nil, err
	}

	result := new(Parameter)
	if err := resolver.Resolve(&ref, result); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveResponse resolves response a reference against a context root
func ResolveResponse(root interface{}, ref Ref) (*Response, error) {
	resolver, err := defaultSchemaLoader(root, nil, nil)
	if err != nil {
		return nil, err
	}

	result := new(Response)
	if err := resolver.Resolve(&ref, result); err != nil {
		return nil, err
	}
	return result, nil
}

type schemaLoader struct {
	loadingRef  *Ref
	startingRef *Ref
	currentRef  *Ref
	root        interface{}
	cache       ResolutionCache
	loadDoc     func(string) (json.RawMessage, error)
}

var idPtr, _ = jsonpointer.New("/id")
var schemaPtr, _ = jsonpointer.New("/$schema")
var refPtr, _ = jsonpointer.New("/$ref")

func defaultSchemaLoader(root interface{}, ref *Ref, cache ResolutionCache) (*schemaLoader, error) {
	if cache == nil {
		cache = resCache
	}

	var ptr *jsonpointer.Pointer
	if ref != nil {
		ptr = ref.GetPointer()
	}

	currentRef := nextRef(root, ref, ptr)

	return &schemaLoader{
		root:        root,
		loadingRef:  ref,
		startingRef: ref,
		cache:       cache,
		loadDoc: func(path string) (json.RawMessage, error) {
			data, err := swag.LoadFromFileOrHTTP(path)
			if err != nil {
				return nil, err
			}
			return json.RawMessage(data), nil
		},
		currentRef: currentRef,
	}, nil
}

func idFromNode(node interface{}) (*Ref, error) {
	if idValue, _, err := idPtr.Get(node); err == nil {
		if refStr, ok := idValue.(string); ok && refStr != "" {
			idRef, err := NewRef(refStr)
			if err != nil {
				return nil, err
			}
			return &idRef, nil
		}
	}
	return nil, nil
}

func nextRef(startingNode interface{}, startingRef *Ref, ptr *jsonpointer.Pointer) *Ref {
	if startingRef == nil {
		return nil
	}
	if ptr == nil {
		return startingRef
	}

	ret := startingRef
	var idRef *Ref
	node := startingNode

	for _, tok := range ptr.DecodedTokens() {
		node, _, _ = jsonpointer.GetForToken(node, tok)
		if node == nil {
			break
		}

		idRef, _ = idFromNode(node)
		if idRef != nil {
			nw, err := ret.Inherits(*idRef)
			if err != nil {
				break
			}
			ret = nw
		}

		refRef, _, _ := refPtr.Get(node)
		if refRef != nil {
			rf, _ := NewRef(refRef.(string))
			nw, err := ret.Inherits(rf)
			if err != nil {
				break
			}
			ret = nw
		}

	}
	return ret
}

func (r *schemaLoader) resolveRef(currentRef, ref *Ref, node, target interface{}) error {
	tgt := reflect.ValueOf(target)
	if tgt.Kind() != reflect.Ptr {
		return fmt.Errorf("resolve ref: target needs to be a pointer")
	}

	oldRef := currentRef
	if currentRef != nil {
		var err error
		currentRef, err = currentRef.Inherits(*nextRef(node, ref, currentRef.GetPointer()))
		if err != nil {
			return err
		}
	}
	if currentRef == nil {
		currentRef = ref
	}

	refURL := currentRef.GetURL()
	if refURL == nil {
		return nil
	}
	if currentRef.IsRoot() {
		nv := reflect.ValueOf(node)
		reflect.Indirect(tgt).Set(reflect.Indirect(nv))
		return nil
	}

	if strings.HasPrefix(refURL.String(), "#") {
		res, _, err := ref.GetPointer().Get(node)
		if err != nil {
			res, _, err = ref.GetPointer().Get(r.root)
			if err != nil {
				return err
			}
		}
		rv := reflect.Indirect(reflect.ValueOf(res))
		tgtType := reflect.Indirect(tgt).Type()
		if rv.Type().AssignableTo(tgtType) {
			reflect.Indirect(tgt).Set(reflect.Indirect(reflect.ValueOf(res)))
		} else {
			if err := swag.DynamicJSONToStruct(rv.Interface(), target); err != nil {
				return err
			}
		}

		return nil
	}

	if refURL.Scheme != "" && refURL.Host != "" {
		// most definitely take the red pill
		data, _, _, err := r.load(refURL)
		if err != nil {
			return err
		}

		if ((oldRef == nil && currentRef != nil) ||
			(oldRef != nil && currentRef == nil) ||
			oldRef.String() != currentRef.String()) &&
			((oldRef == nil && ref != nil) ||
				(oldRef != nil && ref == nil) ||
				(oldRef.String() != ref.String())) {

			return r.resolveRef(currentRef, ref, data, target)
		}

		var res interface{}
		if currentRef.String() != "" {
			res, _, err = currentRef.GetPointer().Get(data)
			if err != nil {
				return err
			}
		} else {
			res = data
		}

		if err := swag.DynamicJSONToStruct(res, target); err != nil {
			return err
		}

	}
	return nil
}

func (r *schemaLoader) load(refURL *url.URL) (interface{}, url.URL, bool, error) {
	toFetch := *refURL
	toFetch.Fragment = ""

	data, fromCache := r.cache.Get(toFetch.String())
	if !fromCache {
		b, err := r.loadDoc(toFetch.String())
		if err != nil {
			return nil, url.URL{}, false, err
		}

		if err := json.Unmarshal(b, &data); err != nil {
			return nil, url.URL{}, false, err
		}
		r.cache.Set(toFetch.String(), data)
	}

	return data, toFetch, fromCache, nil
}
func (r *schemaLoader) Resolve(ref *Ref, target interface{}) error {
	if err := r.resolveRef(r.currentRef, ref, r.root, target); err != nil {
		return err
	}

	return nil
}

type specExpander struct {
	spec     *Swagger
	resolver *schemaLoader
}

// ExpandSpec expands the references in a swagger spec
func ExpandSpec(spec *Swagger) error {
	resolver, err := defaultSchemaLoader(spec, nil, nil)
	if err != nil {
		return err
	}

	for key, defintition := range spec.Definitions {
		var def *Schema
		var err error
		if def, err = expandSchema(defintition, []string{"#/definitions/" + key}, resolver); err != nil {
			return err
		}
		spec.Definitions[key] = *def
	}

	for key, parameter := range spec.Parameters {
		if err := expandParameter(&parameter, resolver); err != nil {
			return err
		}
		spec.Parameters[key] = parameter
	}

	for key, response := range spec.Responses {
		if err := expandResponse(&response, resolver); err != nil {
			return err
		}
		spec.Responses[key] = response
	}

	if spec.Paths != nil {
		for key, path := range spec.Paths.Paths {
			if err := expandPathItem(&path, resolver); err != nil {
				return err
			}
			spec.Paths.Paths[key] = path
		}
	}

	return nil
}

// ExpandSchema expands the refs in the schema object
func ExpandSchema(schema *Schema, root interface{}, cache ResolutionCache) error {

	if schema == nil {
		return nil
	}
	if root == nil {
		root = schema
	}

	nrr, _ := NewRef(schema.ID)
	var rrr *Ref
	if nrr.String() != "" {
		switch root.(type) {
		case *Schema:
			rid, _ := NewRef(root.(*Schema).ID)
			rrr, _ = rid.Inherits(nrr)
		case *Swagger:
			rid, _ := NewRef(root.(*Swagger).ID)
			rrr, _ = rid.Inherits(nrr)
		}

	}

	resolver, err := defaultSchemaLoader(root, rrr, cache)
	if err != nil {
		return err
	}

	refs := []string{""}
	if rrr != nil {
		refs[0] = rrr.String()
	}
	var s *Schema
	if s, err = expandSchema(*schema, refs, resolver); err != nil {
		return nil
	}
	*schema = *s
	return nil
}

func expandItems(target Schema, parentRefs []string, resolver *schemaLoader) (*Schema, error) {
	if target.Items != nil {
		if target.Items.Schema != nil {
			t, err := expandSchema(*target.Items.Schema, parentRefs, resolver)
			if err != nil {
				return nil, err
			}
			*target.Items.Schema = *t
		}
		for i := range target.Items.Schemas {
			t, err := expandSchema(target.Items.Schemas[i], parentRefs, resolver)
			if err != nil {
				return nil, err
			}
			target.Items.Schemas[i] = *t
		}
	}
	return &target, nil
}

func expandSchema(target Schema, parentRefs []string, resolver *schemaLoader) (schema *Schema, err error) {
	defer func() {
		schema = &target
	}()
	if target.Ref.String() == "" && target.Ref.IsRoot() {
		target = *resolver.root.(*Schema)
		return
	}

	// t is the new expanded schema
	var t *Schema
	for target.Ref.String() != "" {
		// var newTarget Schema
		pRefs := strings.Join(parentRefs, ",")
		pRefs += ","
		if strings.Contains(pRefs, target.Ref.String()+",") {
			err = nil
			return
		}

		if err = resolver.Resolve(&target.Ref, &t); err != nil {
			return
		}
		parentRefs = append(parentRefs, target.Ref.String())
		target = *t
	}

	if t, err = expandItems(target, parentRefs, resolver); err != nil {
		return
	}
	target = *t

	for i := range target.AllOf {
		if t, err = expandSchema(target.AllOf[i], parentRefs, resolver); err != nil {
			return
		}
		target.AllOf[i] = *t
	}
	for i := range target.AnyOf {
		if t, err = expandSchema(target.AnyOf[i], parentRefs, resolver); err != nil {
			return
		}
		target.AnyOf[i] = *t
	}
	for i := range target.OneOf {
		if t, err = expandSchema(target.OneOf[i], parentRefs, resolver); err != nil {
			return
		}
		target.OneOf[i] = *t
	}
	if target.Not != nil {
		if t, err = expandSchema(*target.Not, parentRefs, resolver); err != nil {
			return
		}
		*target.Not = *t
	}
	for k, _ := range target.Properties {
		if t, err = expandSchema(target.Properties[k], parentRefs, resolver); err != nil {
			return
		}
		target.Properties[k] = *t
	}
	if target.AdditionalProperties != nil && target.AdditionalProperties.Schema != nil {
		if t, err = expandSchema(*target.AdditionalProperties.Schema, parentRefs, resolver); err != nil {
			return
		}
		*target.AdditionalProperties.Schema = *t
	}
	for k, _ := range target.PatternProperties {
		if t, err = expandSchema(target.PatternProperties[k], parentRefs, resolver); err != nil {
			return
		}
		target.PatternProperties[k] = *t
	}
	for k, _ := range target.Dependencies {
		if target.Dependencies[k].Schema != nil {
			if t, err = expandSchema(*target.Dependencies[k].Schema, parentRefs, resolver); err != nil {
				return
			}
			*target.Dependencies[k].Schema = *t
		}
	}
	if target.AdditionalItems != nil && target.AdditionalItems.Schema != nil {
		if t, err = expandSchema(*target.AdditionalItems.Schema, parentRefs, resolver); err != nil {
			return
		}
		*target.AdditionalItems.Schema = *t
	}
	for k, _ := range target.Definitions {
		if t, err = expandSchema(target.Definitions[k], parentRefs, resolver); err != nil {
			return
		}
		target.Definitions[k] = *t
	}
	return
}

func expandPathItem(pathItem *PathItem, resolver *schemaLoader) error {
	if pathItem == nil {
		return nil
	}
	if pathItem.Ref.String() != "" {
		if err := resolver.Resolve(&pathItem.Ref, &pathItem); err != nil {
			return err
		}
	}

	for idx := range pathItem.Parameters {
		if err := expandParameter(&(pathItem.Parameters[idx]), resolver); err != nil {
			return err
		}
	}
	if err := expandOperation(pathItem.Get, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Head, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Options, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Put, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Post, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Patch, resolver); err != nil {
		return err
	}
	if err := expandOperation(pathItem.Delete, resolver); err != nil {
		return err
	}
	return nil
}

func expandOperation(op *Operation, resolver *schemaLoader) error {
	if op == nil {
		return nil
	}
	for i, param := range op.Parameters {
		if err := expandParameter(&param, resolver); err != nil {
			return err
		}
		op.Parameters[i] = param
	}

	if op.Responses != nil {
		responses := op.Responses
		if err := expandResponse(responses.Default, resolver); err != nil {
			return err
		}
		for code, response := range responses.StatusCodeResponses {
			if err := expandResponse(&response, resolver); err != nil {
				return err
			}
			responses.StatusCodeResponses[code] = response
		}
	}
	return nil
}

func expandResponse(response *Response, resolver *schemaLoader) error {
	if response == nil {
		return nil
	}

	if response.Ref.String() != "" {
		if err := resolver.Resolve(&response.Ref, response); err != nil {
			return err
		}
	}

	if response.Schema != nil {
		parentRefs := []string{response.Schema.Ref.String()}
		if err := resolver.Resolve(&response.Schema.Ref, &response.Schema); err != nil {
			return err
		}
		if s, err := expandSchema(*response.Schema, parentRefs, resolver); err != nil {
			return err
		} else {
			*response.Schema = *s
		}
	}
	return nil
}

func expandParameter(parameter *Parameter, resolver *schemaLoader) error {
	if parameter == nil {
		return nil
	}
	if parameter.Ref.String() != "" {
		if err := resolver.Resolve(&parameter.Ref, parameter); err != nil {
			return err
		}
	}
	if parameter.Schema != nil {
		parentRefs := []string{parameter.Schema.Ref.String()}
		if err := resolver.Resolve(&parameter.Schema.Ref, &parameter.Schema); err != nil {
			return err
		}
		if s, err := expandSchema(*parameter.Schema, parentRefs, resolver); err != nil {
			return err
		} else {
			*parameter.Schema = *s
		}
	}
	return nil
}
