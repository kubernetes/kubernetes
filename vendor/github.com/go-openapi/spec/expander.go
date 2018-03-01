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
	"log"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/swag"
)

var (
	// Debug enables logging when SWAGGER_DEBUG env var is not empty
	Debug = os.Getenv("SWAGGER_DEBUG") != ""
)

// ExpandOptions provides options for expand.
type ExpandOptions struct {
	RelativeBase    string
	SkipSchemas     bool
	ContinueOnError bool
}

// ResolutionCache a cache for resolving urls
type ResolutionCache interface {
	Get(string) (interface{}, bool)
	Set(string, interface{})
}

type simpleCache struct {
	lock  sync.Mutex
	store map[string]interface{}
}

var resCache ResolutionCache

func init() {
	resCache = initResolutionCache()
}

func initResolutionCache() ResolutionCache {
	return &simpleCache{store: map[string]interface{}{
		"http://swagger.io/v2/schema.json":       MustLoadSwagger20Schema(),
		"http://json-schema.org/draft-04/schema": MustLoadJSONSchemaDraft04(),
	}}
}

func (s *simpleCache) Get(uri string) (interface{}, bool) {
	debugLog("getting %q from resolution cache", uri)
	s.lock.Lock()
	v, ok := s.store[uri]
	debugLog("got %q from resolution cache: %t", uri, ok)

	s.lock.Unlock()
	return v, ok
}

func (s *simpleCache) Set(uri string, data interface{}) {
	s.lock.Lock()
	s.store[uri] = data
	s.lock.Unlock()
}

// ResolveRefWithBase resolves a reference against a context root with preservation of base path
func ResolveRefWithBase(root interface{}, ref *Ref, opts *ExpandOptions) (*Schema, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil)
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
		json.Unmarshal(b, newSch)
		return newSch, nil
	default:
		return nil, fmt.Errorf("unknown type for the resolved reference")
	}
}

// ResolveParameter resolves a paramter reference against a context root
func ResolveParameter(root interface{}, ref Ref) (*Parameter, error) {
	return ResolveParameterWithBase(root, ref, nil)
}

// ResolveParameterWithBase resolves a paramter reference against a context root and base path
func ResolveParameterWithBase(root interface{}, ref Ref, opts *ExpandOptions) (*Parameter, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil)
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
	resolver, err := defaultSchemaLoader(root, opts, nil)
	if err != nil {
		return nil, err
	}

	result := new(Response)
	if err := resolver.Resolve(&ref, result, ""); err != nil {
		return nil, err
	}
	return result, nil
}

// ResolveItems resolves header and parameter items reference against a context root and base path
func ResolveItems(root interface{}, ref Ref, opts *ExpandOptions) (*Items, error) {
	resolver, err := defaultSchemaLoader(root, opts, nil)
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
	resolver, err := defaultSchemaLoader(root, opts, nil)
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

type schemaLoader struct {
	root    interface{}
	options *ExpandOptions
	cache   ResolutionCache
	loadDoc func(string) (json.RawMessage, error)
}

var idPtr, _ = jsonpointer.New("/id")
var refPtr, _ = jsonpointer.New("/$ref")

// PathLoader function to use when loading remote refs
var PathLoader func(string) (json.RawMessage, error)

func init() {
	PathLoader = func(path string) (json.RawMessage, error) {
		data, err := swag.LoadFromFileOrHTTP(path)
		if err != nil {
			return nil, err
		}
		return json.RawMessage(data), nil
	}
}

func defaultSchemaLoader(
	root interface{},
	expandOptions *ExpandOptions,
	cache ResolutionCache) (*schemaLoader, error) {

	if cache == nil {
		cache = resCache
	}
	if expandOptions == nil {
		expandOptions = &ExpandOptions{}
	}

	return &schemaLoader{
		root:    root,
		options: expandOptions,
		cache:   cache,
		loadDoc: func(path string) (json.RawMessage, error) {
			debugLog("fetching document at %q", path)
			return PathLoader(path)
		},
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
			var rf Ref
			switch value := refRef.(type) {
			case string:
				rf, _ = NewRef(value)
			}
			nw, err := ret.Inherits(rf)
			if err != nil {
				break
			}
			nwURL := nw.GetURL()
			if nwURL.Scheme == "file" || (nwURL.Scheme == "" && nwURL.Host == "") {
				nwpt := filepath.ToSlash(nwURL.Path)
				if filepath.IsAbs(nwpt) {
					_, err := os.Stat(nwpt)
					if err != nil {
						nwURL.Path = filepath.Join(".", nwpt)
					}
				}
			}

			ret = nw
		}

	}

	return ret
}

func debugLog(msg string, args ...interface{}) {
	if Debug {
		log.Printf(msg, args...)
	}
}

// normalize absolute path for cache.
// on Windows, drive letters should be converted to lower as scheme in net/url.URL
func normalizeAbsPath(path string) string {
	u, err := url.Parse(path)
	if err != nil {
		debugLog("normalize absolute path failed: %s", err)
		return path
	}
	return u.String()
}

// base or refPath could be a file path or a URL
// given a base absolute path and a ref path, return the absolute path of refPath
// 1) if refPath is absolute, return it
// 2) if refPath is relative, join it with basePath keeping the scheme, hosts, and ports if exists
// base could be a directory or a full file path
func normalizePaths(refPath, base string) string {
	refURL, _ := url.Parse(refPath)
	if path.IsAbs(refURL.Path) {
		// refPath is actually absolute
		if refURL.Host != "" {
			return refPath
		}
		return filepath.FromSlash(refPath)
	}

	// relative refPath
	baseURL, _ := url.Parse(base)
	if !strings.HasPrefix(refPath, "#") {
		// combining paths
		if baseURL.Host != "" {
			baseURL.Path = path.Join(path.Dir(baseURL.Path), refURL.Path)
		} else { // base is a file
			newBase := fmt.Sprintf("%s#%s", filepath.Join(filepath.Dir(base), filepath.FromSlash(refURL.Path)), refURL.Fragment)
			return newBase
		}

	}
	// copying fragment from ref to base
	baseURL.Fragment = refURL.Fragment
	return baseURL.String()
}

// relativeBase could be an ABSOLUTE file path or an ABSOLUTE URL
func normalizeFileRef(ref *Ref, relativeBase string) *Ref {
	// This is important for when the reference is pointing to the root schema
	if ref.String() == "" {
		r, _ := NewRef(relativeBase)
		return &r
	}

	refURL := ref.GetURL()
	debugLog("normalizing %s against %s (%s)", ref.String(), relativeBase, refURL.String())

	s := normalizePaths(ref.String(), relativeBase)
	r, _ := NewRef(s)
	return &r
}

func (r *schemaLoader) resolveRef(ref *Ref, target interface{}, basePath string) error {
	tgt := reflect.ValueOf(target)
	if tgt.Kind() != reflect.Ptr {
		return fmt.Errorf("resolve ref: target needs to be a pointer")
	}

	refURL := ref.GetURL()
	if refURL == nil {
		return nil
	}

	var res interface{}
	var data interface{}
	var err error
	// Resolve against the root if it isn't nil, and if ref is pointing at the root, or has a fragment only which means
	// it is pointing somewhere in the root.
	root := r.root
	if (ref.IsRoot() || ref.HasFragmentOnly) && root == nil && basePath != "" {
		if baseRef, err := NewRef(basePath); err == nil {
			root, _, _, _ = r.load(baseRef.GetURL())
		}
	}
	if (ref.IsRoot() || ref.HasFragmentOnly) && root != nil {
		data = root
	} else {
		baseRef := normalizeFileRef(ref, basePath)
		debugLog("current ref is: %s", ref.String())
		debugLog("current ref normalized file: %s", baseRef.String())
		data, _, _, err = r.load(baseRef.GetURL())
		if err != nil {
			return err
		}
	}

	res = data
	if ref.String() != "" {
		res, _, err = ref.GetPointer().Get(data)
		if err != nil {
			return err
		}
	}
	if err := swag.DynamicJSONToStruct(res, target); err != nil {
		return err
	}

	return nil
}

func (r *schemaLoader) load(refURL *url.URL) (interface{}, url.URL, bool, error) {
	debugLog("loading schema from url: %s", refURL)
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

// Resolve resolves a reference against basePath and stores the result in target
// Resolve is not in charge of following references, it only resolves ref by following its URL
// if the schema that ref is referring to has more refs in it. Resolve doesn't resolve them
// if basePath is an empty string, ref is resolved against the root schema stored in the schemaLoader struct
func (r *schemaLoader) Resolve(ref *Ref, target interface{}, basePath string) error {
	return r.resolveRef(ref, target, basePath)
}

// absPath returns the absolute path of a file
func absPath(fname string) (string, error) {
	if strings.HasPrefix(fname, "http") {
		return fname, nil
	}
	if filepath.IsAbs(fname) {
		return fname, nil
	}
	wd, err := os.Getwd()
	return filepath.Join(wd, fname), err
}

// ExpandSpec expands the references in a swagger spec
func ExpandSpec(spec *Swagger, options *ExpandOptions) error {
	resolver, err := defaultSchemaLoader(spec, options, nil)
	// Just in case this ever returns an error.
	if shouldStopOnError(err, resolver.options) {
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
			if def, err = expandSchema(definition, []string{fmt.Sprintf("#/defintions/%s", key)}, resolver, specBasePath); shouldStopOnError(err, resolver.options) {
				return err
			}
			if def != nil {
				spec.Definitions[key] = *def
			}
		}
	}

	for key, parameter := range spec.Parameters {
		if err := expandParameter(&parameter, resolver, specBasePath); shouldStopOnError(err, resolver.options) {
			return err
		}
		spec.Parameters[key] = parameter
	}

	for key, response := range spec.Responses {
		if err := expandResponse(&response, resolver, specBasePath); shouldStopOnError(err, resolver.options) {
			return err
		}
		spec.Responses[key] = response
	}

	if spec.Paths != nil {
		for key, path := range spec.Paths.Paths {
			if err := expandPathItem(&path, resolver, specBasePath); shouldStopOnError(err, resolver.options) {
				return err
			}
			spec.Paths.Paths[key] = path
		}
	}

	return nil
}

func shouldStopOnError(err error, opts *ExpandOptions) bool {
	if err != nil && !opts.ContinueOnError {
		return true
	}

	if err != nil {
		log.Println(err)
	}

	return false
}

// ExpandSchema expands the refs in the schema object with reference to the root object
// go-openapi/validate uses this function
// notice that it is impossible to reference a json scema in a different file other than root
func ExpandSchema(schema *Schema, root interface{}, cache ResolutionCache) error {
	// Only save the root to a tmp file if it isn't nil.
	var base string
	if root != nil {
		base, _ = absPath("root")
		if cache == nil {
			cache = resCache
		}
		cache.Set(normalizeAbsPath(base), root)
		base = "root"
	}

	opts := &ExpandOptions{
		RelativeBase:    base,
		SkipSchemas:     false,
		ContinueOnError: false,
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

	resolver, err := defaultSchemaLoader(nil, opts, cache)
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

// basePathFromSchemaID returns a new basePath based on an existing basePath and a schema ID
func basePathFromSchemaID(oldBasePath, id string) string {
	u, err := url.Parse(oldBasePath)
	if err != nil {
		panic(err)
	}
	uid, err := url.Parse(id)
	if err != nil {
		panic(err)
	}

	if path.IsAbs(uid.Path) {
		return id
	}
	u.Path = path.Join(path.Dir(u.Path), uid.Path)
	return u.String()
}

func isCircular(ref *Ref, basePath string, parentRefs ...string) bool {
	return basePath != "" && swag.ContainsStringsCI(parentRefs, ref.String())
}

func expandSchema(target Schema, parentRefs []string, resolver *schemaLoader, basePath string) (*Schema, error) {
	if target.Ref.String() == "" && target.Ref.IsRoot() {
		// normalizing is important
		newRef := normalizeFileRef(&target.Ref, basePath)
		target.Ref = *newRef
		return &target, nil

	}

	/* change the base path of resolution when an ID is encountered
	   otherwise the basePath should inherit the parent's */
	// important: ID can be relative path
	if target.ID != "" {
		// handling the case when id is a folder
		// remember that basePath has to be a file
		refPath := target.ID
		if strings.HasSuffix(target.ID, "/") {
			// path.Clean here would not work correctly if basepath is http
			refPath = fmt.Sprintf("%s%s", refPath, "placeholder.json")
		}
		basePath = normalizePaths(refPath, basePath)
	}

	/* Explain here what this function does */

	var t *Schema
	/* if Ref is found, everything else doesn't matter */
	/* Ref also changes the resolution scope of children expandSchema */
	if target.Ref.String() != "" {
		/* Here the resolution scope is changed because a $ref was encountered */
		normalizedRef := normalizeFileRef(&target.Ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		/* this means there is a circle in the recursion tree */
		/* return the Ref */
		if isCircular(normalizedRef, basePath, parentRefs...) {
			target.Ref = *normalizedRef
			return &target, nil
		}

		debugLog("\nbasePath: %s", basePath)
		if Debug {
			b, _ := json.Marshal(target)
			debugLog("calling Resolve with target: %s", string(b))
		}
		if err := resolver.Resolve(&target.Ref, &t, basePath); shouldStopOnError(err, resolver.options) {
			return nil, err
		}

		if t != nil {
			parentRefs = append(parentRefs, normalizedRef.String())
			return expandSchema(*t, parentRefs, resolver, normalizedBasePath)
		}
	}

	t, err := expandItems(target, parentRefs, resolver, basePath)
	if shouldStopOnError(err, resolver.options) {
		return &target, err
	}
	if t != nil {
		target = *t
	}

	for i := range target.AllOf {
		t, err := expandSchema(target.AllOf[i], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		target.AllOf[i] = *t
	}
	for i := range target.AnyOf {
		t, err := expandSchema(target.AnyOf[i], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		target.AnyOf[i] = *t
	}
	for i := range target.OneOf {
		t, err := expandSchema(target.OneOf[i], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			target.OneOf[i] = *t
		}
	}
	if target.Not != nil {
		t, err := expandSchema(*target.Not, parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			*target.Not = *t
		}
	}
	for k := range target.Properties {
		t, err := expandSchema(target.Properties[k], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			target.Properties[k] = *t
		}
	}
	if target.AdditionalProperties != nil && target.AdditionalProperties.Schema != nil {
		t, err := expandSchema(*target.AdditionalProperties.Schema, parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			*target.AdditionalProperties.Schema = *t
		}
	}
	for k := range target.PatternProperties {
		t, err := expandSchema(target.PatternProperties[k], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			target.PatternProperties[k] = *t
		}
	}
	for k := range target.Dependencies {
		if target.Dependencies[k].Schema != nil {
			t, err := expandSchema(*target.Dependencies[k].Schema, parentRefs, resolver, basePath)
			if shouldStopOnError(err, resolver.options) {
				return &target, err
			}
			if t != nil {
				*target.Dependencies[k].Schema = *t
			}
		}
	}
	if target.AdditionalItems != nil && target.AdditionalItems.Schema != nil {
		t, err := expandSchema(*target.AdditionalItems.Schema, parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			*target.AdditionalItems.Schema = *t
		}
	}
	for k := range target.Definitions {
		t, err := expandSchema(target.Definitions[k], parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return &target, err
		}
		if t != nil {
			target.Definitions[k] = *t
		}
	}
	return &target, nil
}

func derefPathItem(pathItem *PathItem, parentRefs []string, resolver *schemaLoader, basePath string) error {
	curRef := pathItem.Ref.String()
	if curRef != "" {
		normalizedRef := normalizeFileRef(&pathItem.Ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		if isCircular(normalizedRef, basePath, parentRefs...) {
			return nil
		}

		if err := resolver.Resolve(&pathItem.Ref, pathItem, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}

		if pathItem.Ref.String() != "" && pathItem.Ref.String() != curRef && basePath != normalizedBasePath {
			parentRefs = append(parentRefs, normalizedRef.String())
			return derefPathItem(pathItem, parentRefs, resolver, normalizedBasePath)
		}
	}

	return nil
}

func expandPathItem(pathItem *PathItem, resolver *schemaLoader, basePath string) error {
	if pathItem == nil {
		return nil
	}

	parentRefs := []string{}
	if err := derefPathItem(pathItem, parentRefs, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	pathItem.Ref = Ref{}

	parentRefs = parentRefs[0:]

	for idx := range pathItem.Parameters {
		if err := expandParameter(&(pathItem.Parameters[idx]), resolver, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}
	}
	if err := expandOperation(pathItem.Get, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Head, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Options, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Put, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Post, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Patch, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	if err := expandOperation(pathItem.Delete, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	return nil
}

func expandOperation(op *Operation, resolver *schemaLoader, basePath string) error {
	if op == nil {
		return nil
	}

	for i, param := range op.Parameters {
		if err := expandParameter(&param, resolver, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}
		op.Parameters[i] = param
	}

	if op.Responses != nil {
		responses := op.Responses
		if err := expandResponse(responses.Default, resolver, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}
		for code, response := range responses.StatusCodeResponses {
			if err := expandResponse(&response, resolver, basePath); shouldStopOnError(err, resolver.options) {
				return err
			}
			responses.StatusCodeResponses[code] = response
		}
	}
	return nil
}

// ExpandResponse expands a response based on a basepath
// This is the exported version of expandResponse
// all refs inside response will be resolved relative to basePath
func ExpandResponse(response *Response, basePath string) error {
	opts := &ExpandOptions{
		RelativeBase: basePath,
	}
	resolver, err := defaultSchemaLoader(nil, opts, nil)
	if err != nil {
		return err
	}

	return expandResponse(response, resolver, basePath)
}

func derefResponse(response *Response, parentRefs []string, resolver *schemaLoader, basePath string) error {
	curRef := response.Ref.String()
	if curRef != "" {
		/* Here the resolution scope is changed because a $ref was encountered */
		normalizedRef := normalizeFileRef(&response.Ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		if isCircular(normalizedRef, basePath, parentRefs...) {
			return nil
		}

		if err := resolver.Resolve(&response.Ref, response, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}

		if response.Ref.String() != "" && response.Ref.String() != curRef && basePath != normalizedBasePath {
			parentRefs = append(parentRefs, normalizedRef.String())
			return derefResponse(response, parentRefs, resolver, normalizedBasePath)
		}
	}

	return nil
}

func expandResponse(response *Response, resolver *schemaLoader, basePath string) error {
	if response == nil {
		return nil
	}

	parentRefs := []string{}
	if err := derefResponse(response, parentRefs, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	response.Ref = Ref{}

	parentRefs = parentRefs[0:]
	if !resolver.options.SkipSchemas && response.Schema != nil {
		parentRefs = append(parentRefs, response.Schema.Ref.String())
		s, err := expandSchema(*response.Schema, parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return err
		}
		*response.Schema = *s
	}

	return nil
}

// ExpandParameter expands a parameter based on a basepath
// This is the exported version of expandParameter
// all refs inside parameter will be resolved relative to basePath
func ExpandParameter(parameter *Parameter, basePath string) error {
	opts := &ExpandOptions{
		RelativeBase: basePath,
	}
	resolver, err := defaultSchemaLoader(nil, opts, nil)
	if err != nil {
		return err
	}

	return expandParameter(parameter, resolver, basePath)
}

func derefParameter(parameter *Parameter, parentRefs []string, resolver *schemaLoader, basePath string) error {
	curRef := parameter.Ref.String()
	if curRef != "" {
		normalizedRef := normalizeFileRef(&parameter.Ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		if isCircular(normalizedRef, basePath, parentRefs...) {
			return nil
		}

		if err := resolver.Resolve(&parameter.Ref, parameter, basePath); shouldStopOnError(err, resolver.options) {
			return err
		}

		if parameter.Ref.String() != "" && parameter.Ref.String() != curRef && basePath != normalizedBasePath {
			parentRefs = append(parentRefs, normalizedRef.String())
			return derefParameter(parameter, parentRefs, resolver, normalizedBasePath)
		}
	}

	return nil
}

func expandParameter(parameter *Parameter, resolver *schemaLoader, basePath string) error {
	if parameter == nil {
		return nil
	}

	parentRefs := []string{}
	if err := derefParameter(parameter, parentRefs, resolver, basePath); shouldStopOnError(err, resolver.options) {
		return err
	}
	parameter.Ref = Ref{}

	parentRefs = parentRefs[0:]
	if !resolver.options.SkipSchemas && parameter.Schema != nil {
		parentRefs = append(parentRefs, parameter.Schema.Ref.String())
		s, err := expandSchema(*parameter.Schema, parentRefs, resolver, basePath)
		if shouldStopOnError(err, resolver.options) {
			return err
		}
		*parameter.Schema = *s
	}
	return nil
}
