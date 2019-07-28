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
	"strings"
)

// ExpandOptions provides options for spec expand
type ExpandOptions struct {
	RelativeBase        string
	SkipSchemas         bool
	ContinueOnError     bool
	AbsoluteCircularRef bool
}

// ResolveRefWithBase resolves a reference against a context root with preservation of base path
func ResolveRefWithBase(root interface{}, ref *Ref, opts *ExpandOptions) (*Schema, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return nil, err
	}
	specBasePath := ""
	if opts != nil && opts.RelativeBase != "" {
		specBasePath, _ = absPath(opts.RelativeBase)
	}

	result := new(Schema)
	if err := resolver.Resolve(ref, result, specBasePath); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveRef resolves a reference against a context root
// ref is guaranteed to be in root (no need to go to external files)
// ResolveRef is ONLY called from the code generation module
func ResolveRef(root interface{}, ref *Ref) (*Schema, error) {
	res, _, err := ref.GetPointer().Get(root)
	if err != nil {
		panic(err)
	}
	switch sch := res.(type) {
	case Schema:
		return &sch, nil
	case *Schema:
		return sch, nil
	case map[string]interface{}:
		b, _ := json.Marshal(sch)
		newSch := new(Schema)
		_ = json.Unmarshal(b, newSch)
		return newSch, nil
	default:
		return nil, fmt.Errorf("unknown type for the resolved reference")
	}
}

// ResolveParameter resolves a parameter reference against a context root
func ResolveParameter(root interface{}, ref Ref) (*Parameter, error) {
	return ResolveParameterWithBase(root, ref, nil)
}

// ResolveParameterWithBase resolves a parameter reference against a context root and base path
func ResolveParameterWithBase(root interface{}, ref Ref, opts *ExpandOptions) (*Parameter, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return nil, err
	}

	result := new(Parameter)
	if err := resolver.Resolve(&ref, result, ""); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveResponse resolves response a reference against a context root
func ResolveResponse(root interface{}, ref Ref) (*Response, error) {
	return ResolveResponseWithBase(root, ref, nil)
}

// ResolveResponseWithBase resolves response a reference against a context root and base path
func ResolveResponseWithBase(root interface{}, ref Ref, opts *ExpandOptions) (*Response, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return nil, err
	}

	result := new(Response)
	if err := resolver.Resolve(&ref, result, ""); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveItems resolves parameter items reference against a context root and base path.
//
// NOTE: stricly speaking, this construct is not supported by Swagger 2.0.
// Similarly, $ref are forbidden in response headers.
func ResolveItems(root interface{}, ref Ref, opts *ExpandOptions) (*Items, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return nil, err
	}
	basePath := ""
	if opts.RelativeBase != "" {
		basePath = opts.RelativeBase
	}
	result := new(Items)
	if err := resolver.Resolve(&ref, result, basePath); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolvePathItem resolves response a path item against a context root and base path
func ResolvePathItem(root interface{}, ref Ref, opts *ExpandOptions) (*PathItem, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return nil, err
	}
	basePath := ""
	if opts.RelativeBase != "" {
		basePath = opts.RelativeBase
	}
	result := new(PathItem)
	if err := resolver.Resolve(&ref, result, basePath); err != nil {
		return nil, err
	}
	return result, nil
}

// ExpandSpec expands the references in a swagger spec
func ExpandSpec(spec *Swagger, options *ExpandOptions) error {
	resolver, err := defaultSchemaLoader(spec, options, nil, nil)
	// Just in case this ever returns an error.
	if resolver.shouldStopOnError(err) {
		return err
	}

	// getting the base path of the spec to adjust all subsequent reference resolutions
	specBasePath := ""
	if options != nil && options.RelativeBase != "" {
		specBasePath, _ = absPath(options.RelativeBase)
	}

	if options == nil || !options.SkipSchemas {
		for key, definition := range spec.Definitions {
			var def *Schema
			var err error
			if def, err = expandSchema(definition, []string{fmt.Sprintf("#/definitions/%s", key)}, resolver, specBasePath); resolver.shouldStopOnError(err) {
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
			path := spec.Paths.Paths[key]
			if err := expandPathItem(&path, resolver, specBasePath); resolver.shouldStopOnError(err) {
				return err
			}
			spec.Paths.Paths[key] = path
		}
	}

	return nil
}

// baseForRoot loads in the cache the root document and produces a fake "root" base path entry
// for further $ref resolution
func baseForRoot(root interface{}, cache ResolutionCache) string {
	// cache the root document to resolve $ref's
	const rootBase = "root"
	if root != nil {
		base, _ := absPath(rootBase)
		normalizedBase := normalizeAbsPath(base)
		debugLog("setting root doc in cache at: %s", normalizedBase)
		if cache == nil {
			cache = resCache
		}
		cache.Set(normalizedBase, root)
		return rootBase
	}
	return ""
}

// ExpandSchema expands the refs in the schema object with reference to the root object
// go-openapi/validate uses this function
// notice that it is impossible to reference a json schema in a different file other than root
func ExpandSchema(schema *Schema, root interface{}, cache ResolutionCache) error {
	opts := &ExpandOptions{
		// when a root is specified, cache the root as an in-memory document for $ref retrieval
		RelativeBase:    baseForRoot(root, cache),
		SkipSchemas:     false,
		ContinueOnError: false,
		// when no base path is specified, remaining $ref (circular) are rendered with an absolute path
		AbsoluteCircularRef: true,
	}
	return ExpandSchemaWithBasePath(schema, cache, opts)
}

// ExpandSchemaWithBasePath expands the refs in the schema object, base path configured through expand options
func ExpandSchemaWithBasePath(schema *Schema, cache ResolutionCache, opts *ExpandOptions) error {
	if schema == nil {
		return nil
	}

	var basePath string
	if opts.RelativeBase != "" {
		basePath, _ = absPath(opts.RelativeBase)
	}

	resolver, err := defaultSchemaLoader(nil, opts, cache, nil)
	if err != nil {
		return err
	}

	refs := []string{""}
	var s *Schema
	if s, err = expandSchema(*schema, refs, resolver, basePath); err != nil {
		return err
	}
	*schema = *s
	return nil
}

func expandItems(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	if target.Items != nil {
		if target.Items.Schema != nil {
			t, err := expandSchema(*target.Items.Schema, parentRefs, resolver, basePath)
			if err != nil {
				return nil, err
			}
			*target.Items.Schema = *t
		}
		for i := range target.Items.Schemas {
			t, err := expandSchema(target.Items.Schemas[i], parentRefs, resolver, basePath)
			if err != nil {
				return nil, err
			}
			target.Items.Schemas[i] = *t
		}
	}
	return &target, nil
}

func expandSchema(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	if target.Ref.String() == "" && target.Ref.IsRoot() {
		// normalizing is important
		newRef := normalizeFileRef(&target.Ref, basePath)
		target.Ref = *newRef
		return &target, nil

	}

	// change the base path of resolution when an ID is encountered
	// otherwise the basePath should inherit the parent's
	// important: ID can be relative path
	if target.ID != "" {
		debugLog("schema has ID: %s", target.ID)
		// handling the case when id is a folder
		// remember that basePath has to be a file
		refPath := target.ID
		if strings.HasSuffix(target.ID, "/") {
			// path.Clean here would not work correctly if basepath is http
			refPath = fmt.Sprintf("%s%s", refPath, "placeholder.json")
		}
		basePath = normalizePaths(refPath, basePath)
	}

	var t *Schema
	// if Ref is found, everything else doesn't matter
	// Ref also changes the resolution scope of children expandSchema
	if target.Ref.String() != "" {
		// here the resolution scope is changed because a $ref was encountered
		normalizedRef := normalizeFileRef(&target.Ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		if resolver.isCircular(normalizedRef, basePath, parentRefs...) {
			// this means there is a cycle in the recursion tree: return the Ref
			// - circular refs cannot be expanded. We leave them as ref.
			// - denormalization means that a new local file ref is set relative to the original basePath
			debugLog("shortcut circular ref: basePath: %s, normalizedPath: %s, normalized ref: %s",
				basePath, normalizedBasePath, normalizedRef.String())
			if !resolver.options.AbsoluteCircularRef {
				target.Ref = *denormalizeFileRef(normalizedRef, normalizedBasePath, resolver.context.basePath)
			} else {
				target.Ref = *normalizedRef
			}
			return &target, nil
		}

		debugLog("basePath: %s: calling Resolve with target: %#v", basePath, target)
		if err := resolver.Resolve(&target.Ref, &t, basePath); resolver.shouldStopOnError(err) {
			return nil, err
		}

		if t != nil {
			parentRefs = append(parentRefs, normalizedRef.String())
			var err error
			transitiveResolver, err := resolver.transitiveResolver(basePath, target.Ref)
			if transitiveResolver.shouldStopOnError(err) {
				return nil, err
			}

			basePath = resolver.updateBasePath(transitiveResolver, normalizedBasePath)

			return expandSchema(*t, parentRefs, transitiveResolver, basePath)
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
		target.AllOf[i] = *t
	}
	for i := range target.AnyOf {
		t, err := expandSchema(target.AnyOf[i], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		target.AnyOf[i] = *t
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
	for k := range target.Definitions {
		t, err := expandSchema(target.Definitions[k], parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return &target, err
		}
		if t != nil {
			target.Definitions[k] = *t
		}
	}
	return &target, nil
}

func expandPathItem(pathItem *PathItem, resolver *schemaLoader, basePath string) error {
	if pathItem == nil {
		return nil
	}

	parentRefs := []string{}
	if err := resolver.deref(pathItem, parentRefs, basePath); resolver.shouldStopOnError(err) {
		return err
	}
	if pathItem.Ref.String() != "" {
		var err error
		resolver, err = resolver.transitiveResolver(basePath, pathItem.Ref)
		if resolver.shouldStopOnError(err) {
			return err
		}
	}
	pathItem.Ref = Ref{}

	for idx := range pathItem.Parameters {
		if err := expandParameterOrResponse(&(pathItem.Parameters[idx]), resolver, basePath); resolver.shouldStopOnError(err) {
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

	if op.Responses != nil {
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
	}
	return nil
}

// ExpandResponseWithRoot expands a response based on a root document, not a fetchable document
func ExpandResponseWithRoot(response *Response, root interface{}, cache ResolutionCache) error {
	opts := &ExpandOptions{
		RelativeBase:    baseForRoot(root, cache),
		SkipSchemas:     false,
		ContinueOnError: false,
		// when no base path is specified, remaining $ref (circular) are rendered with an absolute path
		AbsoluteCircularRef: true,
	}
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return err
	}

	return expandParameterOrResponse(response, resolver, opts.RelativeBase)
}

// ExpandResponse expands a response based on a basepath
// This is the exported version of expandResponse
// all refs inside response will be resolved relative to basePath
func ExpandResponse(response *Response, basePath string) error {
	var specBasePath string
	if basePath != "" {
		specBasePath, _ = absPath(basePath)
	}
	opts := &ExpandOptions{
		RelativeBase: specBasePath,
	}
	resolver, err := defaultSchemaLoader(nil, opts, nil, nil)
	if err != nil {
		return err
	}

	return expandParameterOrResponse(response, resolver, opts.RelativeBase)
}

// ExpandParameterWithRoot expands a parameter based on a root document, not a fetchable document
func ExpandParameterWithRoot(parameter *Parameter, root interface{}, cache ResolutionCache) error {
	opts := &ExpandOptions{
		RelativeBase:    baseForRoot(root, cache),
		SkipSchemas:     false,
		ContinueOnError: false,
		// when no base path is specified, remaining $ref (circular) are rendered with an absolute path
		AbsoluteCircularRef: true,
	}
	resolver, err := defaultSchemaLoader(root, opts, nil, nil)
	if err != nil {
		return err
	}

	return expandParameterOrResponse(parameter, resolver, opts.RelativeBase)
}

// ExpandParameter expands a parameter based on a basepath.
// This is the exported version of expandParameter
// all refs inside parameter will be resolved relative to basePath
func ExpandParameter(parameter *Parameter, basePath string) error {
	var specBasePath string
	if basePath != "" {
		specBasePath, _ = absPath(basePath)
	}
	opts := &ExpandOptions{
		RelativeBase: specBasePath,
	}
	resolver, err := defaultSchemaLoader(nil, opts, nil, nil)
	if err != nil {
		return err
	}

	return expandParameterOrResponse(parameter, resolver, opts.RelativeBase)
}

func getRefAndSchema(input interface{}) (*Ref, *Schema, error) {
	var ref *Ref
	var sch *Schema
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
		return nil, nil, fmt.Errorf("expand: unsupported type %T. Input should be of type *Parameter or *Response", input)
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
	parentRefs := []string{}
	if err := resolver.deref(input, parentRefs, basePath); resolver.shouldStopOnError(err) {
		return err
	}
	ref, sch, _ := getRefAndSchema(input)
	if ref.String() != "" {
		transitiveResolver, err := resolver.transitiveResolver(basePath, *ref)
		if transitiveResolver.shouldStopOnError(err) {
			return err
		}
		basePath = resolver.updateBasePath(transitiveResolver, basePath)
		resolver = transitiveResolver
	}

	if sch != nil && sch.Ref.String() != "" {
		// schema expanded to a $ref in another root
		var ern error
		sch.Ref, ern = NewRef(normalizePaths(sch.Ref.String(), ref.RemoteURI()))
		if ern != nil {
			return ern
		}
	}
	if ref != nil {
		*ref = Ref{}
	}

	if !resolver.options.SkipSchemas && sch != nil {
		s, err := expandSchema(*sch, parentRefs, resolver, basePath)
		if resolver.shouldStopOnError(err) {
			return err
		}
		*sch = *s
	}
	return nil
}
