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
)

// ExpandOptions provides options for the spec expander.
//
// RelativeBase is the path to the root document. This can be a remote URL or a path to a local file.
//
// If left empty, the root document is assumed to be located in the current working directory:
// all relative $ref's will be resolved from there.
//
// PathLoader injects a document loading method. By default, this resolves to the function provided by the SpecLoader package variable.
//
type ExpandOptions struct {
	RelativeBase        string                                // the path to the root document to expand. This is a file, not a directory
	SkipSchemas         bool                                  // do not expand schemas, just paths, parameters and responses
	ContinueOnError     bool                                  // continue expanding even after and error is found
	PathLoader          func(string) (json.RawMessage, error) `json:"-"` // the document loading method that takes a path as input and yields a json document
	AbsoluteCircularRef bool                                  // circular $ref remaining after expansion remain absolute URLs
}

func optionsOrDefault(opts *ExpandOptions) *ExpandOptions {
	if opts != nil {
		clone := *opts // shallow clone to avoid internal changes to be propagated to the caller
		if clone.RelativeBase != "" {
			clone.RelativeBase = normalizeBase(clone.RelativeBase)
		}
		// if the relative base is empty, let the schema loader choose a pseudo root document
		return &clone
	}
	return &ExpandOptions{}
}

// ExpandSpec expands the references in a swagger spec
func ExpandSpec(spec *Swagger, options *ExpandOptions) error {
	options = optionsOrDefault(options)
	resolver := defaultSchemaLoader(spec, options, nil, nil)

	specBasePath := options.RelativeBase

	if !options.SkipSchemas {
		for key, definition := range spec.Definitions {
			parentRefs := make([]string, 0, 10)
			parentRefs = append(parentRefs, fmt.Sprintf("#/definitions/%s", key))

			def, err := expandSchema(definition, parentRefs, resolver, specBasePath)
			if resolver.shouldStopOnError(err) {
				return err
			}
			if def != nil {
				spec.Definitions[key] = *def
			}
		}
	}

	for key := range spec.Parameters {
		parameter := spec.Parameters[key]
		if err := expandParameterOrResponse(&parameter, resolver, specBasePath); resolver.shouldStopOnError(err) {
			return err
		}
		spec.Parameters[key] = parameter
	}

	for key := range spec.Responses {
		response := spec.Responses[key]
		if err := expandParameterOrResponse(&response, resolver, specBasePath); resolver.shouldStopOnError(err) {
			return err
		}
		spec.Responses[key] = response
	}

	if spec.Paths != nil {
		for key := range spec.Paths.Paths {
			pth := spec.Paths.Paths[key]
			if err := expandPathItem(&pth, resolver, specBasePath); resolver.shouldStopOnError(err) {
				return err
			}
			spec.Paths.Paths[key] = pth
		}
	}

	return nil
}

const rootBase = ".root"

// baseForRoot loads in the cache the root document and produces a fake ".root" base path entry
// for further $ref resolution
//
// Setting the cache is optional and this parameter may safely be left to nil.
func baseForRoot(root interface{}, cache ResolutionCache) string {
	if root == nil {
		return ""
	}

	// cache the root document to resolve $ref's
	normalizedBase := normalizeBase(rootBase)
	cache.Set(normalizedBase, root)

	return normalizedBase
}

// ExpandSchema expands the refs in the schema object with reference to the root object.
//
// go-openapi/validate uses this function.
//
// Notice that it is impossible to reference a json schema in a different document other than root
// (use ExpandSchemaWithBasePath to resolve external references).
//
// Setting the cache is optional and this parameter may safely be left to nil.
func ExpandSchema(schema *Schema, root interface{}, cache ResolutionCache) error {
	cache = cacheOrDefault(cache)
	if root == nil {
		root = schema
	}

	opts := &ExpandOptions{
		// when a root is specified, cache the root as an in-memory document for $ref retrieval
		RelativeBase:    baseForRoot(root, cache),
		SkipSchemas:     false,
		ContinueOnError: false,
	}

	return ExpandSchemaWithBasePath(schema, cache, opts)
}

// ExpandSchemaWithBasePath expands the refs in the schema object, base path configured through expand options.
//
// Setting the cache is optional and this parameter may safely be left to nil.
func ExpandSchemaWithBasePath(schema *Schema, cache ResolutionCache, opts *ExpandOptions) error {
	if schema == nil {
		return nil
	}

	cache = cacheOrDefault(cache)

	opts = optionsOrDefault(opts)

	resolver := defaultSchemaLoader(nil, opts, cache, nil)

	parentRefs := make([]string, 0, 10)
	s, err := expandSchema(*schema, parentRefs, resolver, opts.RelativeBase)
	if err != nil {
		return err
	}
	if s != nil {
		// guard for when continuing on error
		*schema = *s
	}

	return nil
}

func expandItems(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	if target.Items == nil {
		return &target, nil
	}

	// array
	if target.Items.Schema != nil {
		t, err := expandSchema(*target.Items.Schema, parentRefs, resolver, basePath)
		if err != nil {
			return nil, err
		}
		*target.Items.Schema = *t
	}

	// tuple
	for i := range target.Items.Schemas {
		t, err := expandSchema(target.Items.Schemas[i], parentRefs, resolver, basePath)
		if err != nil {
			return nil, err
		}
		target.Items.Schemas[i] = *t
	}

	return &target, nil
}

func expandSchema(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	if target.Ref.String() == "" && target.Ref.IsRoot() {
		newRef := normalizeRef(&target.Ref, basePath)
		target.Ref = *newRef
		return &target, nil
	}

	// change the base path of resolution when an ID is encountered
	// otherwise the basePath should inherit the parent's
	if target.ID != "" {
		basePath, _ = resolver.setSchemaID(target, target.ID, basePath)
	}

	if target.Ref.String() != "" {
		return expandSchemaRef(target, parentRefs, resolver, basePath)
	}

	for k := range target.Definitions {
		tt, err := expandSchema(target.Definitions[k], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if tt != nil {
			target.Definitions[k] = *tt
		}
	}

	t, err := expandItems(target, parentRefs, resolver, basePath)
	if resolver.shouldStopOnError(err) {
		return &target, err
	}
	if t != nil {
		target = *t
	}

	for i := range target.AllOf {
		t, err := expandSchema(target.AllOf[i], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.AllOf[i] = *t
		}
	}

	for i := range target.AnyOf {
		t, err := expandSchema(target.AnyOf[i], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.AnyOf[i] = *t
		}
	}

	for i := range target.OneOf {
		t, err := expandSchema(target.OneOf[i], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.OneOf[i] = *t
		}
	}

	if target.Not != nil {
		t, err := expandSchema(*target.Not, parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			*target.Not = *t
		}
	}

	for k := range target.Properties {
		t, err := expandSchema(target.Properties[k], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.Properties[k] = *t
		}
	}

	if target.AdditionalProperties != nil && target.AdditionalProperties.Schema != nil {
		t, err := expandSchema(*target.AdditionalProperties.Schema, parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			*target.AdditionalProperties.Schema = *t
		}
	}

	for k := range target.PatternProperties {
		t, err := expandSchema(target.PatternProperties[k], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.PatternProperties[k] = *t
		}
	}

	for k := range target.Dependencies {
		if target.Dependencies[k].Schema != nil {
			t, err := expandSchema(*target.Dependencies[k].Schema, parentRefs, resolver, basePath)
			if resolver.shouldStopOnError(err) {
				return &target, err
			}
			if t != nil {
				*target.Dependencies[k].Schema = *t
			}
		}
	}

	if target.AdditionalItems != nil && target.AdditionalItems.Schema != nil {
		t, err := expandSchema(*target.AdditionalItems.Schema, parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			*target.AdditionalItems.Schema = *t
		}
	}
	return &target, nil
}

func expandSchemaRef(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	// if a Ref is found, all sibling fields are skipped
	// Ref also changes the resolution scope of children expandSchema

	// here the resolution scope is changed because a $ref was encountered
	normalizedRef := normalizeRef(&target.Ref, basePath)
	normalizedBasePath := normalizedRef.RemoteURI()

	if resolver.isCircular(normalizedRef, basePath, parentRefs...) {
		// this means there is a cycle in the recursion tree: return the Ref
		// - circular refs cannot be expanded. We leave them as ref.
		// - denormalization means that a new local file ref is set relative to the original basePath
		debugLog("short circuit circular ref: basePath: %s, normalizedPath: %s, normalized ref: %s",
			basePath, normalizedBasePath, normalizedRef.String())
		if !resolver.options.AbsoluteCircularRef {
			target.Ref = denormalizeRef(normalizedRef, resolver.context.basePath, resolver.context.rootID)
		} else {
			target.Ref = *normalizedRef
		}
		return &target, nil
	}

	var t *Schema
	err := resolver.Resolve(&target.Ref, &t, basePath)
	if resolver.shouldStopOnError(err) {
		return nil, err
	}

	if t == nil {
		// guard for when continuing on error
		return &target, nil
	}

	parentRefs = append(parentRefs, normalizedRef.String())
	transitiveResolver := resolver.transitiveResolver(basePath, target.Ref)

	basePath = resolver.updateBasePath(transitiveResolver, normalizedBasePath)

	return expandSchema(*t, parentRefs, transitiveResolver, basePath)
}

func expandPathItem(pathItem *PathItem, resolver *schemaLoader, basePath string) error {
	if pathItem == nil {
		return nil
	}

	parentRefs := make([]string, 0, 10)
	if err := resolver.deref(pathItem, parentRefs, basePath); resolver.shouldStopOnError(err) {
		return err
	}

	if pathItem.Ref.String() != "" {
		transitiveResolver := resolver.transitiveResolver(basePath, pathItem.Ref)
		basePath = transitiveResolver.updateBasePath(resolver, basePath)
		resolver = transitiveResolver
	}

	pathItem.Ref = Ref{}
	for i := range pathItem.Parameters {
		if err := expandParameterOrResponse(&(pathItem.Parameters[i]), resolver, basePath); resolver.shouldStopOnError(err) {
			return err
		}
	}

	ops := []*Operation{
		pathItem.Get,
		pathItem.Head,
		pathItem.Options,
		pathItem.Put,
		pathItem.Post,
		pathItem.Patch,
		pathItem.Delete,
	}
	for _, op := range ops {
		if err := expandOperation(op, resolver, basePath); resolver.shouldStopOnError(err) {
			return err
		}
	}

	return nil
}

func expandOperation(op *Operation, resolver *schemaLoader, basePath string) error {
	if op == nil {
		return nil
	}

	for i := range op.Parameters {
		param := op.Parameters[i]
		if err := expandParameterOrResponse(&param, resolver, basePath); resolver.shouldStopOnError(err) {
			return err
		}
		op.Parameters[i] = param
	}

	if op.Responses == nil {
		return nil
	}

	responses := op.Responses
	if err := expandParameterOrResponse(responses.Default, resolver, basePath); resolver.shouldStopOnError(err) {
		return err
	}

	for code := range responses.StatusCodeResponses {
		response := responses.StatusCodeResponses[code]
		if err := expandParameterOrResponse(&response, resolver, basePath); resolver.shouldStopOnError(err) {
			return err
		}
		responses.StatusCodeResponses[code] = response
	}

	return nil
}

// ExpandResponseWithRoot expands a response based on a root document, not a fetchable document
//
// Notice that it is impossible to reference a json schema in a different document other than root
// (use ExpandResponse to resolve external references).
//
// Setting the cache is optional and this parameter may safely be left to nil.
func ExpandResponseWithRoot(response *Response, root interface{}, cache ResolutionCache) error {
	cache = cacheOrDefault(cache)
	opts := &ExpandOptions{
		RelativeBase: baseForRoot(root, cache),
	}
	resolver := defaultSchemaLoader(root, opts, cache, nil)

	return expandParameterOrResponse(response, resolver, opts.RelativeBase)
}

// ExpandResponse expands a response based on a basepath
//
// All refs inside response will be resolved relative to basePath
func ExpandResponse(response *Response, basePath string) error {
	opts := optionsOrDefault(&ExpandOptions{
		RelativeBase: basePath,
	})
	resolver := defaultSchemaLoader(nil, opts, nil, nil)

	return expandParameterOrResponse(response, resolver, opts.RelativeBase)
}

// ExpandParameterWithRoot expands a parameter based on a root document, not a fetchable document.
//
// Notice that it is impossible to reference a json schema in a different document other than root
// (use ExpandParameter to resolve external references).
func ExpandParameterWithRoot(parameter *Parameter, root interface{}, cache ResolutionCache) error {
	cache = cacheOrDefault(cache)

	opts := &ExpandOptions{
		RelativeBase: baseForRoot(root, cache),
	}
	resolver := defaultSchemaLoader(root, opts, cache, nil)

	return expandParameterOrResponse(parameter, resolver, opts.RelativeBase)
}

// ExpandParameter expands a parameter based on a basepath.
// This is the exported version of expandParameter
// all refs inside parameter will be resolved relative to basePath
func ExpandParameter(parameter *Parameter, basePath string) error {
	opts := optionsOrDefault(&ExpandOptions{
		RelativeBase: basePath,
	})
	resolver := defaultSchemaLoader(nil, opts, nil, nil)

	return expandParameterOrResponse(parameter, resolver, opts.RelativeBase)
}

func getRefAndSchema(input interface{}) (*Ref, *Schema, error) {
	var (
		ref *Ref
		sch *Schema
	)

	switch refable := input.(type) {
	case *Parameter:
		if refable == nil {
			return nil, nil, nil
		}
		ref = &refable.Ref
		sch = refable.Schema
	case *Response:
		if refable == nil {
			return nil, nil, nil
		}
		ref = &refable.Ref
		sch = refable.Schema
	default:
		return nil, nil, fmt.Errorf("unsupported type: %T: %w", input, ErrExpandUnsupportedType)
	}

	return ref, sch, nil
}

func expandParameterOrResponse(input interface{}, resolver *schemaLoader, basePath string) error {
	ref, _, err := getRefAndSchema(input)
	if err != nil {
		return err
	}

	if ref == nil {
		return nil
	}

	parentRefs := make([]string, 0, 10)
	if err = resolver.deref(input, parentRefs, basePath); resolver.shouldStopOnError(err) {
		return err
	}

	ref, sch, _ := getRefAndSchema(input)
	if ref.String() != "" {
		transitiveResolver := resolver.transitiveResolver(basePath, *ref)
		basePath = resolver.updateBasePath(transitiveResolver, basePath)
		resolver = transitiveResolver
	}

	if sch == nil {
		// nothing to be expanded
		if ref != nil {
			*ref = Ref{}
		}
		return nil
	}

	if sch.Ref.String() != "" {
		rebasedRef, ern := NewRef(normalizeURI(sch.Ref.String(), basePath))
		if ern != nil {
			return ern
		}

		switch {
		case resolver.isCircular(&rebasedRef, basePath, parentRefs...):
			// this is a circular $ref: stop expansion
			if !resolver.options.AbsoluteCircularRef {
				sch.Ref = denormalizeRef(&rebasedRef, resolver.context.basePath, resolver.context.rootID)
			} else {
				sch.Ref = rebasedRef
			}
		case !resolver.options.SkipSchemas:
			// schema expanded to a $ref in another root
			sch.Ref = rebasedRef
			debugLog("rebased to: %s", sch.Ref.String())
		default:
			// skip schema expansion but rebase $ref to schema
			sch.Ref = denormalizeRef(&rebasedRef, resolver.context.basePath, resolver.context.rootID)
		}
	}

	if ref != nil {
		*ref = Ref{}
	}

	// expand schema
	if !resolver.options.SkipSchemas {
		s, err := expandSchema(*sch, parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return err
		}
		if s == nil {
			// guard for when continuing on error
			return nil
		}
		*sch = *s
	}

	return nil
}
