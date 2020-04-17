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
	"reflect"
	"strings"

	"github.com/go-openapi/swag"
)

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

// resolverContext allows to share a context during spec processing.
// At the moment, it just holds the index of circular references found.
type resolverContext struct {
	// circulars holds all visited circular references, which allows shortcuts.
	// NOTE: this is not just a performance improvement: it is required to figure out
	// circular references which participate several cycles.
	// This structure is privately instantiated and needs not be locked against
	// concurrent access, unless we chose to implement a parallel spec walking.
	circulars map[string]bool
	basePath  string
}

func newResolverContext(originalBasePath string) *resolverContext {
	return &resolverContext{
		circulars: make(map[string]bool),
		basePath:  originalBasePath, // keep the root base path in context
	}
}

type schemaLoader struct {
	root    interface{}
	options *ExpandOptions
	cache   ResolutionCache
	context *resolverContext
	loadDoc func(string) (json.RawMessage, error)
}

func (r *schemaLoader) transitiveResolver(basePath string, ref Ref) (*schemaLoader, error) {
	if ref.IsRoot() || ref.HasFragmentOnly {
		return r, nil
	}

	baseRef, _ := NewRef(basePath)
	currentRef := normalizeFileRef(&ref, basePath)
	if strings.HasPrefix(currentRef.String(), baseRef.String()) {
		return r, nil
	}

	// Set a new root to resolve against
	rootURL := currentRef.GetURL()
	rootURL.Fragment = ""
	root, _ := r.cache.Get(rootURL.String())

	// shallow copy of resolver options to set a new RelativeBase when
	// traversing multiple documents
	newOptions := r.options
	newOptions.RelativeBase = rootURL.String()
	debugLog("setting new root: %s", newOptions.RelativeBase)
	resolver, err := defaultSchemaLoader(root, newOptions, r.cache, r.context)
	if err != nil {
		return nil, err
	}

	return resolver, nil
}

func (r *schemaLoader) updateBasePath(transitive *schemaLoader, basePath string) string {
	if transitive != r {
		debugLog("got a new resolver")
		if transitive.options != nil && transitive.options.RelativeBase != "" {
			basePath, _ = absPath(transitive.options.RelativeBase)
			debugLog("new basePath = %s", basePath)
		}
	}
	return basePath
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
		if baseRef, erb := NewRef(basePath); erb == nil {
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
	return swag.DynamicJSONToStruct(res, target)
}

func (r *schemaLoader) load(refURL *url.URL) (interface{}, url.URL, bool, error) {
	debugLog("loading schema from url: %s", refURL)
	toFetch := *refURL
	toFetch.Fragment = ""

	normalized := normalizeAbsPath(toFetch.String())

	data, fromCache := r.cache.Get(normalized)
	if !fromCache {
		b, err := r.loadDoc(normalized)
		if err != nil {
			debugLog("unable to load the document: %v", err)
			return nil, url.URL{}, false, err
		}

		if err := json.Unmarshal(b, &data); err != nil {
			return nil, url.URL{}, false, err
		}
		r.cache.Set(normalized, data)
	}

	return data, toFetch, fromCache, nil
}

// isCircular detects cycles in sequences of $ref.
// It relies on a private context (which needs not be locked).
func (r *schemaLoader) isCircular(ref *Ref, basePath string, parentRefs ...string) (foundCycle bool) {
	normalizedRef := normalizePaths(ref.String(), basePath)
	if _, ok := r.context.circulars[normalizedRef]; ok {
		// circular $ref has been already detected in another explored cycle
		foundCycle = true
		return
	}
	foundCycle = swag.ContainsStringsCI(parentRefs, normalizedRef)
	if foundCycle {
		r.context.circulars[normalizedRef] = true
	}
	return
}

// Resolve resolves a reference against basePath and stores the result in target
// Resolve is not in charge of following references, it only resolves ref by following its URL
// if the schema that ref is referring to has more refs in it. Resolve doesn't resolve them
// if basePath is an empty string, ref is resolved against the root schema stored in the schemaLoader struct
func (r *schemaLoader) Resolve(ref *Ref, target interface{}, basePath string) error {
	return r.resolveRef(ref, target, basePath)
}

func (r *schemaLoader) deref(input interface{}, parentRefs []string, basePath string) error {
	var ref *Ref
	switch refable := input.(type) {
	case *Schema:
		ref = &refable.Ref
	case *Parameter:
		ref = &refable.Ref
	case *Response:
		ref = &refable.Ref
	case *PathItem:
		ref = &refable.Ref
	default:
		return fmt.Errorf("deref: unsupported type %T", input)
	}

	curRef := ref.String()
	if curRef != "" {
		normalizedRef := normalizeFileRef(ref, basePath)
		normalizedBasePath := normalizedRef.RemoteURI()

		if r.isCircular(normalizedRef, basePath, parentRefs...) {
			return nil
		}

		if err := r.resolveRef(ref, input, basePath); r.shouldStopOnError(err) {
			return err
		}

		// NOTE(fredbi): removed basePath check => needs more testing
		if ref.String() != "" && ref.String() != curRef {
			parentRefs = append(parentRefs, normalizedRef.String())
			return r.deref(input, parentRefs, normalizedBasePath)
		}
	}

	return nil
}

func (r *schemaLoader) shouldStopOnError(err error) bool {
	if err != nil && !r.options.ContinueOnError {
		return true
	}

	if err != nil {
		log.Println(err)
	}

	return false
}

func defaultSchemaLoader(
	root interface{},
	expandOptions *ExpandOptions,
	cache ResolutionCache,
	context *resolverContext) (*schemaLoader, error) {

	if cache == nil {
		cache = resCache
	}
	if expandOptions == nil {
		expandOptions = &ExpandOptions{}
	}
	absBase, _ := absPath(expandOptions.RelativeBase)
	if context == nil {
		context = newResolverContext(absBase)
	}
	return &schemaLoader{
		root:    root,
		options: expandOptions,
		cache:   cache,
		context: context,
		loadDoc: func(path string) (json.RawMessage, error) {
			debugLog("fetching document at %q", path)
			return PathLoader(path)
		},
	}, nil
}
