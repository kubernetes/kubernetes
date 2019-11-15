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

package analysis

import (
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	slashpath "path"
	"path/filepath"
	"sort"
	"strings"

	"strconv"

	"github.com/go-openapi/analysis/internal"
	"github.com/go-openapi/jsonpointer"
	swspec "github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

// FlattenOpts configuration for flattening a swagger specification.
type FlattenOpts struct {
	Spec           *Spec    // The analyzed spec to work with
	flattenContext *context // Internal context to track flattening activity

	BasePath string

	// Flattening options
	Expand       bool // If Expand is true, we skip flattening the spec and expand it instead
	Minimal      bool
	Verbose      bool
	RemoveUnused bool

	/* Extra keys */
	_ struct{} // require keys
}

// ExpandOpts creates a spec.ExpandOptions to configure expanding a specification document.
func (f *FlattenOpts) ExpandOpts(skipSchemas bool) *swspec.ExpandOptions {
	return &swspec.ExpandOptions{RelativeBase: f.BasePath, SkipSchemas: skipSchemas}
}

// Swagger gets the swagger specification for this flatten operation
func (f *FlattenOpts) Swagger() *swspec.Swagger {
	return f.Spec.spec
}

// newRef stores information about refs created during the flattening process
type newRef struct {
	key      string
	newName  string
	path     string
	isOAIGen bool
	resolved bool
	schema   *swspec.Schema
	parents  []string
}

// context stores intermediary results from flatten
type context struct {
	newRefs  map[string]*newRef
	warnings []string
	resolved map[string]string
}

func newContext() *context {
	return &context{
		newRefs:  make(map[string]*newRef, 150),
		warnings: make([]string, 0),
		resolved: make(map[string]string, 50),
	}
}

// Flatten an analyzed spec and produce a self-contained spec bundle.
//
// There is a minimal and a full flattening mode.
//
// Minimally flattening a spec means:
//  - Expanding parameters, responses, path items, parameter items and header items (references to schemas are left
//    unscathed)
//  - Importing external (http, file) references so they become internal to the document
//  - Moving every JSON pointer to a $ref to a named definition (i.e. the reworked spec does not contain pointers
//    like "$ref": "#/definitions/myObject/allOfs/1")
//
// A minimally flattened spec thus guarantees the following properties:
//  - all $refs point to a local definition (i.e. '#/definitions/...')
//  - definitions are unique
//
// NOTE: arbitrary JSON pointers (other than $refs to top level definitions) are rewritten as definitions if they
// represent a complex schema or express commonality in the spec.
// Otherwise, they are simply expanded.
//
// Minimal flattening is necessary and sufficient for codegen rendering using go-swagger.
//
// Fully flattening a spec means:
//  - Moving every complex inline schema to be a definition with an auto-generated name in a depth-first fashion.
//
// By complex, we mean every JSON object with some properties.
// Arrays, when they do not define a tuple,
// or empty objects with or without additionalProperties, are not considered complex and remain inline.
//
// NOTE: rewritten schemas get a vendor extension x-go-gen-location so we know from which part of the spec definitions
// have been created.
//
// Available flattening options:
//  - Minimal: stops flattening after minimal $ref processing, leaving schema constructs untouched
//  - Expand: expand all $ref's in the document (inoperant if Minimal set to true)
//  - Verbose: croaks about name conflicts detected
//  - RemoveUnused: removes unused parameters, responses and definitions after expansion/flattening
//
// NOTE: expansion removes all $ref save circular $ref, which remain in place
//
// TODO: additional options
//  - ProgagateNameExtensions: ensure that created entries properly follow naming rules when their parent have set a
//    x-go-name extension
//  - LiftAllOfs:
//     - limit the flattening of allOf members when simple objects
//     - merge allOf with validation only
//     - merge allOf with extensions only
//     - ...
//
func Flatten(opts FlattenOpts) error {
	// Make sure opts.BasePath is an absolute path
	if !filepath.IsAbs(opts.BasePath) {
		cwd, _ := os.Getwd()
		opts.BasePath = filepath.Join(cwd, opts.BasePath)
	}
	// make sure drive letter on windows is normalized to lower case
	u, _ := url.Parse(opts.BasePath)
	opts.BasePath = u.String()

	opts.flattenContext = newContext()

	// recursively expand responses, parameters, path items and items in simple schemas.
	// This simplifies the spec and leaves $ref only into schema objects.
	if err := swspec.ExpandSpec(opts.Swagger(), opts.ExpandOpts(!opts.Expand)); err != nil {
		return err
	}

	// strip current file from $ref's, so we can recognize them as proper definitions
	// In particular, this works around for issue go-openapi/spec#76: leading absolute file in $ref is stripped
	if err := normalizeRef(&opts); err != nil {
		return err
	}

	if opts.RemoveUnused {
		// optionally removes shared parameters and responses already expanded (now unused)
		// default parameters (i.e. under paths) remain.
		opts.Swagger().Parameters = nil
		opts.Swagger().Responses = nil
	}

	opts.Spec.reload() // re-analyze

	// at this point there are no references left but in schemas

	for imported := false; !imported; {
		// iteratively import remote references until none left.
		// This inlining deals with name conflicts by introducing auto-generated names ("OAIGen")
		var err error
		if imported, err = importExternalReferences(&opts); err != nil {
			return err
		}
		opts.Spec.reload() // re-analyze
	}

	if !opts.Minimal && !opts.Expand {
		// full flattening: rewrite inline schemas (schemas that aren't simple types or arrays or maps)
		if err := nameInlinedSchemas(&opts); err != nil {
			return err
		}

		opts.Spec.reload() // re-analyze
	}

	// rewrite JSON pointers other than $ref to named definitions
	// and attempt to resolve conflicting names whenever possible.
	if err := stripPointersAndOAIGen(&opts); err != nil {
		return err
	}

	if opts.RemoveUnused {
		// remove unused definitions
		expected := make(map[string]struct{})
		for k := range opts.Swagger().Definitions {
			expected[slashpath.Join(definitionsPath, jsonpointer.Escape(k))] = struct{}{}
		}
		for _, k := range opts.Spec.AllDefinitionReferences() {
			delete(expected, k)
		}
		for k := range expected {
			debugLog("removing unused definition %s", slashpath.Base(k))
			if opts.Verbose {
				log.Printf("info: removing unused definition: %s", slashpath.Base(k))
			}
			delete(opts.Swagger().Definitions, slashpath.Base(k))
		}
		opts.Spec.reload() // re-analyze
	}

	// TODO: simplify known schema patterns to flat objects with properties
	// examples:
	//  - lift simple allOf object,
	//  - empty allOf with validation only or extensions only
	//  - rework allOf arrays
	//  - rework allOf additionalProperties

	if opts.Verbose {
		// issue notifications
		croak(&opts)
	}
	return nil
}

// isAnalyzedAsComplex determines if an analyzed schema is eligible to flattening (i.e. it is "complex").
//
// Complex means the schema is any of:
//  - a simple type (primitive)
//  - an array of something (items are possibly complex ; if this is the case, items will generate a definition)
//  - a map of something (additionalProperties are possibly complex ; if this is the case, additionalProperties will
//    generate a definition)
func isAnalyzedAsComplex(asch *AnalyzedSchema) bool {
	if !asch.IsSimpleSchema && !asch.IsArray && !asch.IsMap {
		return true
	}
	return false
}

// nameInlinedSchemas replaces every complex inline construct by a named definition.
func nameInlinedSchemas(opts *FlattenOpts) error {
	debugLog("nameInlinedSchemas")
	namer := &inlineSchemaNamer{
		Spec:           opts.Swagger(),
		Operations:     opRefsByRef(gatherOperations(opts.Spec, nil)),
		flattenContext: opts.flattenContext,
		opts:           opts,
	}
	depthFirst := sortDepthFirst(opts.Spec.allSchemas)
	for _, key := range depthFirst {
		sch := opts.Spec.allSchemas[key]
		if sch.Schema != nil && sch.Schema.Ref.String() == "" && !sch.TopLevel { // inline schema
			asch, err := Schema(SchemaOpts{Schema: sch.Schema, Root: opts.Swagger(), BasePath: opts.BasePath})
			if err != nil {
				return fmt.Errorf("schema analysis [%s]: %v", key, err)
			}

			if isAnalyzedAsComplex(asch) { // move complex schemas to definitions
				if err := namer.Name(key, sch.Schema, asch); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

var depthGroupOrder = []string{
	"sharedParam", "sharedResponse", "sharedOpParam", "opParam", "codeResponse", "defaultResponse", "definition",
}

func sortDepthFirst(data map[string]SchemaRef) []string {
	// group by category (shared params, op param, statuscode response, default response, definitions)
	// sort groups internally by number of parts in the key and lexical names
	// flatten groups into a single list of keys
	sorted := make([]string, 0, len(data))
	grouped := make(map[string]keys, len(data))
	for k := range data {
		split := keyParts(k)
		var pk string
		if split.IsSharedOperationParam() {
			pk = "sharedOpParam"
		}
		if split.IsOperationParam() {
			pk = "opParam"
		}
		if split.IsStatusCodeResponse() {
			pk = "codeResponse"
		}
		if split.IsDefaultResponse() {
			pk = "defaultResponse"
		}
		if split.IsDefinition() {
			pk = "definition"
		}
		if split.IsSharedParam() {
			pk = "sharedParam"
		}
		if split.IsSharedResponse() {
			pk = "sharedResponse"
		}
		grouped[pk] = append(grouped[pk], key{Segments: len(split), Key: k})
	}

	for _, pk := range depthGroupOrder {
		res := grouped[pk]
		sort.Sort(res)
		for _, v := range res {
			sorted = append(sorted, v.Key)
		}
	}
	return sorted
}

type key struct {
	Segments int
	Key      string
}
type keys []key

func (k keys) Len() int      { return len(k) }
func (k keys) Swap(i, j int) { k[i], k[j] = k[j], k[i] }
func (k keys) Less(i, j int) bool {
	return k[i].Segments > k[j].Segments || (k[i].Segments == k[j].Segments && k[i].Key < k[j].Key)
}

type inlineSchemaNamer struct {
	Spec           *swspec.Swagger
	Operations     map[string]opRef
	flattenContext *context
	opts           *FlattenOpts
}

func opRefsByRef(oprefs map[string]opRef) map[string]opRef {
	result := make(map[string]opRef, len(oprefs))
	for _, v := range oprefs {
		result[v.Ref.String()] = v
	}
	return result
}

func (isn *inlineSchemaNamer) Name(key string, schema *swspec.Schema, aschema *AnalyzedSchema) error {
	debugLog("naming inlined schema at %s", key)

	parts := keyParts(key)
	for _, name := range namesFromKey(parts, aschema, isn.Operations) {
		if name != "" {
			// create unique name
			newName, isOAIGen := uniqifyName(isn.Spec.Definitions, swag.ToJSONName(name))

			// clone schema
			sch, err := cloneSchema(schema)
			if err != nil {
				return err
			}

			// replace values on schema
			if err := rewriteSchemaToRef(isn.Spec, key,
				swspec.MustCreateRef(slashpath.Join(definitionsPath, newName))); err != nil {
				return fmt.Errorf("error while creating definition %q from inline schema: %v", newName, err)
			}

			// rewrite any dependent $ref pointing to this place,
			// when not already pointing to a top-level definition.
			//
			// NOTE: this is important if such referers use arbitrary JSON pointers.
			an := New(isn.Spec)
			for k, v := range an.references.allRefs {
				r, _, erd := deepestRef(isn.opts, v)
				if erd != nil {
					return fmt.Errorf("at %s, %v", k, erd)
				}
				if r.String() == key ||
					r.String() == slashpath.Join(definitionsPath, newName) &&
						slashpath.Dir(v.String()) != definitionsPath {
					debugLog("found a $ref to a rewritten schema: %s points to %s", k, v.String())

					// rewrite $ref to the new target
					if err := updateRef(isn.Spec, k,
						swspec.MustCreateRef(slashpath.Join(definitionsPath, newName))); err != nil {
						return err
					}
				}
			}

			// NOTE: this extension is currently not used by go-swagger (provided for information only)
			sch.AddExtension("x-go-gen-location", genLocation(parts))

			// save cloned schema to definitions
			saveSchema(isn.Spec, newName, sch)

			// keep track of created refs
			if isn.flattenContext != nil {
				debugLog("track created ref: key=%s, newName=%s, isOAIGen=%t", key, newName, isOAIGen)
				resolved := false
				if _, ok := isn.flattenContext.newRefs[key]; ok {
					resolved = isn.flattenContext.newRefs[key].resolved
				}
				isn.flattenContext.newRefs[key] = &newRef{
					key:      key,
					newName:  newName,
					path:     slashpath.Join(definitionsPath, newName),
					isOAIGen: isOAIGen,
					resolved: resolved,
					schema:   sch,
				}
			}
		}
	}
	return nil
}

// genLocation indicates from which section of the specification (models or operations) a definition has been created.
//
// This is reflected in the output spec with a "x-go-gen-location" extension. At the moment, this is is provided
// for information only.
func genLocation(parts splitKey) string {
	if parts.IsOperation() {
		return "operations"
	}
	if parts.IsDefinition() {
		return "models"
	}
	return ""
}

// uniqifyName yields a unique name for a definition
func uniqifyName(definitions swspec.Definitions, name string) (string, bool) {
	isOAIGen := false
	if name == "" {
		name = "oaiGen"
		isOAIGen = true
	}
	if len(definitions) == 0 {
		return name, isOAIGen
	}

	unq := true
	for k := range definitions {
		if strings.EqualFold(k, name) {
			unq = false
			break
		}
	}

	if unq {
		return name, isOAIGen
	}

	name += "OAIGen"
	isOAIGen = true
	var idx int
	unique := name
	_, known := definitions[unique]
	for known {
		idx++
		unique = fmt.Sprintf("%s%d", name, idx)
		_, known = definitions[unique]
	}
	return unique, isOAIGen
}

func namesFromKey(parts splitKey, aschema *AnalyzedSchema, operations map[string]opRef) []string {
	var baseNames [][]string
	var startIndex int
	if parts.IsOperation() {
		// params
		if parts.IsOperationParam() || parts.IsSharedOperationParam() {
			piref := parts.PathItemRef()
			if piref.String() != "" && parts.IsOperationParam() {
				if op, ok := operations[piref.String()]; ok {
					startIndex = 5
					baseNames = append(baseNames, []string{op.ID, "params", "body"})
				}
			} else if parts.IsSharedOperationParam() {
				pref := parts.PathRef()
				for k, v := range operations {
					if strings.HasPrefix(k, pref.String()) {
						startIndex = 4
						baseNames = append(baseNames, []string{v.ID, "params", "body"})
					}
				}
			}
		}
		// responses
		if parts.IsOperationResponse() {
			piref := parts.PathItemRef()
			if piref.String() != "" {
				if op, ok := operations[piref.String()]; ok {
					startIndex = 6
					baseNames = append(baseNames, []string{op.ID, parts.ResponseName(), "body"})
				}
			}
		}
	}

	// definitions
	if parts.IsDefinition() {
		nm := parts.DefinitionName()
		if nm != "" {
			startIndex = 2
			baseNames = append(baseNames, []string{parts.DefinitionName()})
		}
	}

	var result []string
	for _, segments := range baseNames {
		nm := parts.BuildName(segments, startIndex, aschema)
		if nm != "" {
			result = append(result, nm)
		}
	}
	sort.Strings(result)
	return result
}

const (
	paths           = "paths"
	responses       = "responses"
	parameters      = "parameters"
	definitions     = "definitions"
	definitionsPath = "#/definitions"
)

var (
	ignoredKeys  map[string]struct{}
	validMethods map[string]struct{}
)

func init() {
	ignoredKeys = map[string]struct{}{
		"schema":     {},
		"properties": {},
		"not":        {},
		"anyOf":      {},
		"oneOf":      {},
	}

	validMethods = map[string]struct{}{
		"GET":     {},
		"HEAD":    {},
		"OPTIONS": {},
		"PATCH":   {},
		"POST":    {},
		"PUT":     {},
		"DELETE":  {},
	}
}

type splitKey []string

func (s splitKey) IsDefinition() bool {
	return len(s) > 1 && s[0] == definitions
}

func (s splitKey) DefinitionName() string {
	if !s.IsDefinition() {
		return ""
	}
	return s[1]
}

func (s splitKey) isKeyName(i int) bool {
	if i <= 0 {
		return false
	}
	count := 0
	for idx := i - 1; idx > 0; idx-- {
		if s[idx] != "properties" {
			break
		}
		count++
	}

	return count%2 != 0
}

func (s splitKey) BuildName(segments []string, startIndex int, aschema *AnalyzedSchema) string {
	for i, part := range s[startIndex:] {
		if _, ignored := ignoredKeys[part]; !ignored || s.isKeyName(startIndex+i) {
			if part == "items" || part == "additionalItems" {
				if aschema.IsTuple || aschema.IsTupleWithExtra {
					segments = append(segments, "tuple")
				} else {
					segments = append(segments, "items")
				}
				if part == "additionalItems" {
					segments = append(segments, part)
				}
				continue
			}
			segments = append(segments, part)
		}
	}
	return strings.Join(segments, " ")
}

func (s splitKey) IsOperation() bool {
	return len(s) > 1 && s[0] == paths
}

func (s splitKey) IsSharedOperationParam() bool {
	return len(s) > 2 && s[0] == paths && s[2] == parameters
}

func (s splitKey) IsSharedParam() bool {
	return len(s) > 1 && s[0] == parameters
}

func (s splitKey) IsOperationParam() bool {
	return len(s) > 3 && s[0] == paths && s[3] == parameters
}

func (s splitKey) IsOperationResponse() bool {
	return len(s) > 3 && s[0] == paths && s[3] == responses
}

func (s splitKey) IsSharedResponse() bool {
	return len(s) > 1 && s[0] == responses
}

func (s splitKey) IsDefaultResponse() bool {
	return len(s) > 4 && s[0] == paths && s[3] == responses && s[4] == "default"
}

func (s splitKey) IsStatusCodeResponse() bool {
	isInt := func() bool {
		_, err := strconv.Atoi(s[4])
		return err == nil
	}
	return len(s) > 4 && s[0] == paths && s[3] == responses && isInt()
}

func (s splitKey) ResponseName() string {
	if s.IsStatusCodeResponse() {
		code, _ := strconv.Atoi(s[4])
		return http.StatusText(code)
	}
	if s.IsDefaultResponse() {
		return "Default"
	}
	return ""
}

func (s splitKey) PathItemRef() swspec.Ref {
	if len(s) < 3 {
		return swspec.Ref{}
	}
	pth, method := s[1], s[2]
	if _, isValidMethod := validMethods[strings.ToUpper(method)]; !isValidMethod && !strings.HasPrefix(method, "x-") {
		return swspec.Ref{}
	}
	return swspec.MustCreateRef("#" + slashpath.Join("/", paths, jsonpointer.Escape(pth), strings.ToUpper(method)))
}

func (s splitKey) PathRef() swspec.Ref {
	if !s.IsOperation() {
		return swspec.Ref{}
	}
	return swspec.MustCreateRef("#" + slashpath.Join("/", paths, jsonpointer.Escape(s[1])))
}

func keyParts(key string) splitKey {
	var res []string
	for _, part := range strings.Split(key[1:], "/") {
		if part != "" {
			res = append(res, jsonpointer.Unescape(part))
		}
	}
	return res
}

func rewriteSchemaToRef(spec *swspec.Swagger, key string, ref swspec.Ref) error {
	debugLog("rewriting schema to ref for %s with %s", key, ref.String())
	_, value, err := getPointerFromKey(spec, key)
	if err != nil {
		return err
	}

	switch refable := value.(type) {
	case *swspec.Schema:
		return rewriteParentRef(spec, key, ref)

	case swspec.Schema:
		return rewriteParentRef(spec, key, ref)

	case *swspec.SchemaOrArray:
		if refable.Schema != nil {
			refable.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		}

	case *swspec.SchemaOrBool:
		if refable.Schema != nil {
			refable.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		}
	default:
		return fmt.Errorf("no schema with ref found at %s for %T", key, value)
	}

	return nil
}

func rewriteParentRef(spec *swspec.Swagger, key string, ref swspec.Ref) error {
	parent, entry, pvalue, err := getParentFromKey(spec, key)
	if err != nil {
		return err
	}

	debugLog("rewriting holder for %T", pvalue)
	switch container := pvalue.(type) {
	case swspec.Response:
		if err := rewriteParentRef(spec, "#"+parent, ref); err != nil {
			return err
		}

	case *swspec.Response:
		container.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	case *swspec.Responses:
		statusCode, err := strconv.Atoi(entry)
		if err != nil {
			return fmt.Errorf("%s not a number: %v", key[1:], err)
		}
		resp := container.StatusCodeResponses[statusCode]
		resp.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		container.StatusCodeResponses[statusCode] = resp

	case map[string]swspec.Response:
		resp := container[entry]
		resp.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		container[entry] = resp

	case swspec.Parameter:
		if err := rewriteParentRef(spec, "#"+parent, ref); err != nil {
			return err
		}

	case map[string]swspec.Parameter:
		param := container[entry]
		param.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		container[entry] = param

	case []swspec.Parameter:
		idx, err := strconv.Atoi(entry)
		if err != nil {
			return fmt.Errorf("%s not a number: %v", key[1:], err)
		}
		param := container[idx]
		param.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		container[idx] = param

	case swspec.Definitions:
		container[entry] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	case map[string]swspec.Schema:
		container[entry] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	case []swspec.Schema:
		idx, err := strconv.Atoi(entry)
		if err != nil {
			return fmt.Errorf("%s not a number: %v", key[1:], err)
		}
		container[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	case *swspec.SchemaOrArray:
		// NOTE: this is necessarily an array - otherwise, the parent would be *Schema
		idx, err := strconv.Atoi(entry)
		if err != nil {
			return fmt.Errorf("%s not a number: %v", key[1:], err)
		}
		container.Schemas[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	// NOTE: can't have case *swspec.SchemaOrBool = parent in this case is *Schema

	default:
		return fmt.Errorf("unhandled parent schema rewrite %s (%T)", key, pvalue)
	}
	return nil
}

func cloneSchema(schema *swspec.Schema) (*swspec.Schema, error) {
	var sch swspec.Schema
	if err := swag.FromDynamicJSON(schema, &sch); err != nil {
		return nil, fmt.Errorf("cannot clone schema: %v", err)
	}
	return &sch, nil
}

// importExternalReferences iteratively digs remote references and imports them into the main schema.
//
// At every iteration, new remotes may be found when digging deeper: they are rebased to the current schema before being imported.
//
// This returns true when no more remote references can be found.
func importExternalReferences(opts *FlattenOpts) (bool, error) {
	debugLog("importExternalReferences")

	groupedRefs := reverseIndexForSchemaRefs(opts)
	sortedRefStr := make([]string, 0, len(groupedRefs))
	if opts.flattenContext == nil {
		opts.flattenContext = newContext()
	}

	// sort $ref resolution to ensure deterministic name conflict resolution
	for refStr := range groupedRefs {
		sortedRefStr = append(sortedRefStr, refStr)
	}
	sort.Strings(sortedRefStr)

	complete := true

	for _, refStr := range sortedRefStr {
		entry := groupedRefs[refStr]
		if entry.Ref.HasFragmentOnly {
			continue
		}
		complete = false
		var isOAIGen bool

		newName := opts.flattenContext.resolved[refStr]
		if newName != "" {
			// rewrite ref with already resolved external ref (useful for cyclical refs):
			// rewrite external refs to local ones
			debugLog("resolving known ref [%s] to %s", refStr, newName)
			for _, key := range entry.Keys {
				if err := updateRef(opts.Swagger(), key,
					swspec.MustCreateRef(slashpath.Join(definitionsPath, newName))); err != nil {
					return false, err
				}
			}
		} else {
			// resolve schemas
			debugLog("resolving schema from remote $ref [%s]", refStr)
			sch, err := swspec.ResolveRefWithBase(opts.Swagger(), &entry.Ref, opts.ExpandOpts(false))
			if err != nil {
				return false, fmt.Errorf("could not resolve schema: %v", err)
			}

			// at this stage only $ref analysis matters
			partialAnalyzer := &Spec{
				references: referenceAnalysis{},
				patterns:   patternAnalysis{},
				enums:      enumAnalysis{},
			}
			partialAnalyzer.reset()
			partialAnalyzer.analyzeSchema("", *sch, "/")

			// now rewrite those refs with rebase
			for key, ref := range partialAnalyzer.references.allRefs {
				if err := updateRef(sch, key, swspec.MustCreateRef(rebaseRef(entry.Ref.String(), ref.String()))); err != nil {
					return false, fmt.Errorf("failed to rewrite ref for key %q at %s: %v", key, entry.Ref.String(), err)
				}
			}

			// generate a unique name - isOAIGen means that a naming conflict was resolved by changing the name
			newName, isOAIGen = uniqifyName(opts.Swagger().Definitions, nameFromRef(entry.Ref))
			debugLog("new name for [%s]: %s - with name conflict:%t",
				strings.Join(entry.Keys, ", "), newName, isOAIGen)

			opts.flattenContext.resolved[refStr] = newName

			// rewrite the external refs to local ones
			for _, key := range entry.Keys {
				if err := updateRef(opts.Swagger(), key,
					swspec.MustCreateRef(slashpath.Join(definitionsPath, newName))); err != nil {
					return false, err
				}

				// keep track of created refs
				resolved := false
				if _, ok := opts.flattenContext.newRefs[key]; ok {
					resolved = opts.flattenContext.newRefs[key].resolved
				}
				opts.flattenContext.newRefs[key] = &newRef{
					key:      key,
					newName:  newName,
					path:     slashpath.Join(definitionsPath, newName),
					isOAIGen: isOAIGen,
					resolved: resolved,
					schema:   sch,
				}
			}

			// add the resolved schema to the definitions
			saveSchema(opts.Swagger(), newName, sch)
		}
	}
	// maintains ref index entries
	for k := range opts.flattenContext.newRefs {
		r := opts.flattenContext.newRefs[k]

		// update tracking with resolved schemas
		if r.schema.Ref.String() != "" {
			ref := swspec.MustCreateRef(r.path)
			sch, err := swspec.ResolveRefWithBase(opts.Swagger(), &ref, opts.ExpandOpts(false))
			if err != nil {
				return false, fmt.Errorf("could not resolve schema: %v", err)
			}
			r.schema = sch
		}
		// update tracking with renamed keys: got a cascade of refs
		if r.path != k {
			renamed := *r
			renamed.key = r.path
			opts.flattenContext.newRefs[renamed.path] = &renamed

			// indirect ref
			r.newName = slashpath.Base(k)
			r.schema = swspec.RefSchema(r.path)
			r.path = k
			r.isOAIGen = strings.Contains(k, "OAIGen")
		}
	}

	return complete, nil
}

type refRevIdx struct {
	Ref  swspec.Ref
	Keys []string
}

// rebaseRef rebase a remote ref relative to a base ref.
//
// NOTE: does not support JSONschema ID for $ref (we assume we are working with swagger specs here).
//
// NOTE(windows):
// * refs are assumed to have been normalized with drive letter lower cased (from go-openapi/spec)
// * "/ in paths may appear as escape sequences
func rebaseRef(baseRef string, ref string) string {
	debugLog("rebasing ref: %s onto %s", ref, baseRef)
	baseRef, _ = url.PathUnescape(baseRef)
	ref, _ = url.PathUnescape(ref)
	if baseRef == "" || baseRef == "." || strings.HasPrefix(baseRef, "#") {
		return ref
	}

	parts := strings.Split(ref, "#")

	baseParts := strings.Split(baseRef, "#")
	baseURL, _ := url.Parse(baseParts[0])
	if strings.HasPrefix(ref, "#") {
		if baseURL.Host == "" {
			return strings.Join([]string{baseParts[0], parts[1]}, "#")
		}
		return strings.Join([]string{baseParts[0], parts[1]}, "#")
	}

	refURL, _ := url.Parse(parts[0])
	if refURL.Host != "" || filepath.IsAbs(parts[0]) {
		// not rebasing an absolute path
		return ref
	}

	// there is a relative path
	var basePath string
	if baseURL.Host != "" {
		// when there is a host, standard URI rules apply (with "/")
		baseURL.Path = slashpath.Dir(baseURL.Path)
		baseURL.Path = slashpath.Join(baseURL.Path, "/"+parts[0])
		return baseURL.String()
	}

	// this is a local relative path
	// basePart[0] and parts[0] are local filesystem directories/files
	basePath = filepath.Dir(baseParts[0])
	relPath := filepath.Join(basePath, string(filepath.Separator)+parts[0])
	if len(parts) > 1 {
		return strings.Join([]string{relPath, parts[1]}, "#")
	}
	return relPath
}

// normalizePath renders absolute path on remote file refs
//
// NOTE(windows):
// * refs are assumed to have been normalized with drive letter lower cased (from go-openapi/spec)
// * "/ in paths may appear as escape sequences
func normalizePath(ref swspec.Ref, opts *FlattenOpts) (normalizedPath string) {
	uri, _ := url.PathUnescape(ref.String())
	if ref.HasFragmentOnly || filepath.IsAbs(uri) {
		normalizedPath = uri
		return
	}

	refURL, _ := url.Parse(uri)
	if refURL.Host != "" {
		normalizedPath = uri
		return
	}

	parts := strings.Split(uri, "#")
	// BasePath, parts[0] are local filesystem directories, guaranteed to be absolute at this stage
	parts[0] = filepath.Join(filepath.Dir(opts.BasePath), parts[0])
	normalizedPath = strings.Join(parts, "#")
	return
}

func reverseIndexForSchemaRefs(opts *FlattenOpts) map[string]refRevIdx {
	collected := make(map[string]refRevIdx)
	for key, schRef := range opts.Spec.references.schemas {
		// normalize paths before sorting,
		// so we get together keys in same external file
		normalizedPath := normalizePath(schRef, opts)
		if entry, ok := collected[normalizedPath]; ok {
			entry.Keys = append(entry.Keys, key)
			collected[normalizedPath] = entry
		} else {
			collected[normalizedPath] = refRevIdx{
				Ref:  schRef,
				Keys: []string{key},
			}
		}
	}
	return collected
}

func nameFromRef(ref swspec.Ref) string {
	u := ref.GetURL()
	if u.Fragment != "" {
		return swag.ToJSONName(slashpath.Base(u.Fragment))
	}
	if u.Path != "" {
		bn := slashpath.Base(u.Path)
		if bn != "" && bn != "/" {
			ext := slashpath.Ext(bn)
			if ext != "" {
				return swag.ToJSONName(bn[:len(bn)-len(ext)])
			}
			return swag.ToJSONName(bn)
		}
	}
	return swag.ToJSONName(strings.Replace(u.Host, ".", " ", -1))
}

func saveSchema(spec *swspec.Swagger, name string, schema *swspec.Schema) {
	if schema == nil {
		return
	}
	if spec.Definitions == nil {
		spec.Definitions = make(map[string]swspec.Schema, 150)
	}
	spec.Definitions[name] = *schema
}

// getPointerFromKey retrieves the content of the JSON pointer "key"
func getPointerFromKey(spec interface{}, key string) (string, interface{}, error) {
	switch spec.(type) {
	case *swspec.Schema:
	case *swspec.Swagger:
	default:
		panic("unexpected type used in getPointerFromKey")
	}
	if key == "#/" {
		return "", spec, nil
	}
	// unescape chars in key, e.g. "{}" from path params
	pth, _ := internal.PathUnescape(key[1:])
	ptr, err := jsonpointer.New(pth)
	if err != nil {
		return "", nil, err
	}

	value, _, err := ptr.Get(spec)
	if err != nil {
		debugLog("error when getting key: %s with path: %s", key, pth)
		return "", nil, err
	}
	return pth, value, nil
}

// getParentFromKey retrieves the container of the JSON pointer "key"
func getParentFromKey(spec interface{}, key string) (string, string, interface{}, error) {
	switch spec.(type) {
	case *swspec.Schema:
	case *swspec.Swagger:
	default:
		panic("unexpected type used in getPointerFromKey")
	}
	// unescape chars in key, e.g. "{}" from path params
	pth, _ := internal.PathUnescape(key[1:])

	parent, entry := slashpath.Dir(pth), slashpath.Base(pth)
	debugLog("getting schema holder at: %s, with entry: %s", parent, entry)

	pptr, err := jsonpointer.New(parent)
	if err != nil {
		return "", "", nil, err
	}
	pvalue, _, err := pptr.Get(spec)
	if err != nil {
		return "", "", nil, fmt.Errorf("can't get parent for %s: %v", parent, err)
	}
	return parent, entry, pvalue, nil
}

// updateRef replaces a ref by another one
func updateRef(spec interface{}, key string, ref swspec.Ref) error {
	switch spec.(type) {
	case *swspec.Schema:
	case *swspec.Swagger:
	default:
		panic("unexpected type used in getPointerFromKey")
	}
	debugLog("updating ref for %s with %s", key, ref.String())
	pth, value, err := getPointerFromKey(spec, key)
	if err != nil {
		return err
	}

	switch refable := value.(type) {
	case *swspec.Schema:
		refable.Ref = ref
	case *swspec.SchemaOrArray:
		if refable.Schema != nil {
			refable.Schema.Ref = ref
		}
	case *swspec.SchemaOrBool:
		if refable.Schema != nil {
			refable.Schema.Ref = ref
		}
	case swspec.Schema:
		debugLog("rewriting holder for %T", refable)
		_, entry, pvalue, erp := getParentFromKey(spec, key)
		if erp != nil {
			return err
		}
		switch container := pvalue.(type) {
		case swspec.Definitions:
			container[entry] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

		case map[string]swspec.Schema:
			container[entry] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

		case []swspec.Schema:
			idx, err := strconv.Atoi(entry)
			if err != nil {
				return fmt.Errorf("%s not a number: %v", pth, err)
			}
			container[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

		case *swspec.SchemaOrArray:
			// NOTE: this is necessarily an array - otherwise, the parent would be *Schema
			idx, err := strconv.Atoi(entry)
			if err != nil {
				return fmt.Errorf("%s not a number: %v", pth, err)
			}
			container.Schemas[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

		// NOTE: can't have case *swspec.SchemaOrBool = parent in this case is *Schema

		default:
			return fmt.Errorf("unhandled container type at %s: %T", key, value)
		}

	default:
		return fmt.Errorf("no schema with ref found at %s for %T", key, value)
	}

	return nil
}

// updateRefWithSchema replaces a ref with a schema (i.e. re-inline schema)
func updateRefWithSchema(spec *swspec.Swagger, key string, sch *swspec.Schema) error {
	debugLog("updating ref for %s with schema", key)
	pth, value, err := getPointerFromKey(spec, key)
	if err != nil {
		return err
	}

	switch refable := value.(type) {
	case *swspec.Schema:
		*refable = *sch
	case swspec.Schema:
		_, entry, pvalue, erp := getParentFromKey(spec, key)
		if erp != nil {
			return err
		}
		switch container := pvalue.(type) {
		case swspec.Definitions:
			container[entry] = *sch

		case map[string]swspec.Schema:
			container[entry] = *sch

		case []swspec.Schema:
			idx, err := strconv.Atoi(entry)
			if err != nil {
				return fmt.Errorf("%s not a number: %v", pth, err)
			}
			container[idx] = *sch

		case *swspec.SchemaOrArray:
			// NOTE: this is necessarily an array - otherwise, the parent would be *Schema
			idx, err := strconv.Atoi(entry)
			if err != nil {
				return fmt.Errorf("%s not a number: %v", pth, err)
			}
			container.Schemas[idx] = *sch

		// NOTE: can't have case *swspec.SchemaOrBool = parent in this case is *Schema

		default:
			return fmt.Errorf("unhandled type for parent of [%s]: %T", key, value)
		}
	case *swspec.SchemaOrArray:
		*refable.Schema = *sch
	// NOTE: can't have case *swspec.SchemaOrBool = parent in this case is *Schema
	case *swspec.SchemaOrBool:
		*refable.Schema = *sch
	default:
		return fmt.Errorf("no schema with ref found at %s for %T", key, value)
	}

	return nil
}

func containsString(names []string, name string) bool {
	for _, nm := range names {
		if nm == name {
			return true
		}
	}
	return false
}

type opRef struct {
	Method string
	Path   string
	Key    string
	ID     string
	Op     *swspec.Operation
	Ref    swspec.Ref
}

type opRefs []opRef

func (o opRefs) Len() int           { return len(o) }
func (o opRefs) Swap(i, j int)      { o[i], o[j] = o[j], o[i] }
func (o opRefs) Less(i, j int) bool { return o[i].Key < o[j].Key }

func gatherOperations(specDoc *Spec, operationIDs []string) map[string]opRef {
	var oprefs opRefs

	for method, pathItem := range specDoc.Operations() {
		for pth, operation := range pathItem {
			vv := *operation
			oprefs = append(oprefs, opRef{
				Key:    swag.ToGoName(strings.ToLower(method) + " " + pth),
				Method: method,
				Path:   pth,
				ID:     vv.ID,
				Op:     &vv,
				Ref:    swspec.MustCreateRef("#" + slashpath.Join("/paths", jsonpointer.Escape(pth), method)),
			})
		}
	}

	sort.Sort(oprefs)

	operations := make(map[string]opRef)
	for _, opr := range oprefs {
		nm := opr.ID
		if nm == "" {
			nm = opr.Key
		}

		oo, found := operations[nm]
		if found && oo.Method != opr.Method && oo.Path != opr.Path {
			nm = opr.Key
		}
		if len(operationIDs) == 0 || containsString(operationIDs, opr.ID) || containsString(operationIDs, nm) {
			opr.ID = nm
			opr.Op.ID = nm
			operations[nm] = opr
		}
	}
	return operations
}

// stripPointersAndOAIGen removes anonymous JSON pointers from spec and chain with name conflicts handler.
// This loops until the spec has no such pointer and all name conflicts have been reduced as much as possible.
func stripPointersAndOAIGen(opts *FlattenOpts) error {
	// name all JSON pointers to anonymous documents
	if err := namePointers(opts); err != nil {
		return err
	}

	// remove unnecessary OAIGen ref (created when flattening external refs creates name conflicts)
	hasIntroducedPointerOrInline, ers := stripOAIGen(opts)
	if ers != nil {
		return ers
	}

	// iterate as pointer or OAIGen resolution may introduce inline schemas or pointers
	for hasIntroducedPointerOrInline {
		if !opts.Minimal {
			opts.Spec.reload() // re-analyze
			if err := nameInlinedSchemas(opts); err != nil {
				return err
			}
		}

		if err := namePointers(opts); err != nil {
			return err
		}

		// restrip
		if hasIntroducedPointerOrInline, ers = stripOAIGen(opts); ers != nil {
			return ers
		}

		opts.Spec.reload() // re-analyze
	}
	return nil
}

// stripOAIGen strips the spec from unnecessary OAIGen constructs, initially created to dedupe flattened definitions.
//
// A dedupe is deemed unnecessary whenever:
//  - the only conflict is with its (single) parent: OAIGen is merged into its parent (reinlining)
//  - there is a conflict with multiple parents: merge OAIGen in first parent, the rewrite other parents to point to
//    the first parent.
//
// This function returns a true bool whenever it re-inlined a complex schema, so the caller may chose to iterate
// pointer and name resolution again.
func stripOAIGen(opts *FlattenOpts) (bool, error) {
	debugLog("stripOAIGen")
	replacedWithComplex := false

	// figure out referers of OAIGen definitions
	for _, r := range opts.flattenContext.newRefs {
		if !r.isOAIGen || r.resolved { // bail on already resolved entries (avoid looping)
			continue
		}
		for k, v := range opts.Spec.references.allRefs {
			if r.path != v.String() {
				continue
			}
			found := false
			for _, p := range r.parents {
				if p == k {
					found = true
					break
				}
			}
			if !found {
				r.parents = append(r.parents, k)
			}
		}
	}

	for k := range opts.flattenContext.newRefs {
		r := opts.flattenContext.newRefs[k]
		//debugLog("newRefs[%s]: isOAIGen: %t, resolved: %t, name: %s, path:%s, #parents: %d, parents: %v,  ref: %s",
		//	k, r.isOAIGen, r.resolved, r.newName, r.path, len(r.parents), r.parents, r.schema.Ref.String())
		if r.isOAIGen && len(r.parents) >= 1 {
			pr := r.parents
			sort.Strings(pr)

			// rewrite first parent schema in lexicographical order
			debugLog("rewrite first parent in lex order %s with schema", pr[0])
			if err := updateRefWithSchema(opts.Swagger(), pr[0], r.schema); err != nil {
				return false, err
			}
			if pa, ok := opts.flattenContext.newRefs[pr[0]]; ok && pa.isOAIGen {
				// update parent in ref index entry
				debugLog("update parent entry: %s", pr[0])
				pa.schema = r.schema
				pa.resolved = false
				replacedWithComplex = true
			}

			// rewrite other parents to point to first parent
			if len(pr) > 1 {
				for _, p := range pr[1:] {
					replacingRef := swspec.MustCreateRef(pr[0])

					// set complex when replacing ref is an anonymous jsonpointer: further processing may be required
					replacedWithComplex = replacedWithComplex ||
						slashpath.Dir(replacingRef.String()) != definitionsPath
					debugLog("rewrite parent with ref: %s", replacingRef.String())

					// NOTE: it is possible at this stage to introduce json pointers (to non-definitions places).
					// Those are stripped later on.
					if err := updateRef(opts.Swagger(), p, replacingRef); err != nil {
						return false, err
					}

					if pa, ok := opts.flattenContext.newRefs[p]; ok && pa.isOAIGen {
						// update parent in ref index
						debugLog("update parent entry: %s", p)
						pa.schema = r.schema
						pa.resolved = false
						replacedWithComplex = true
					}
				}
			}

			// remove OAIGen definition
			debugLog("removing definition %s", slashpath.Base(r.path))
			delete(opts.Swagger().Definitions, slashpath.Base(r.path))

			// propagate changes in ref index for keys which have this one as a parent
			for kk, value := range opts.flattenContext.newRefs {
				if kk == k || !value.isOAIGen || value.resolved {
					continue
				}
				found := false
				newParents := make([]string, 0, len(value.parents))
				for _, parent := range value.parents {
					switch {
					case parent == r.path:
						found = true
						parent = pr[0]
					case strings.HasPrefix(parent, r.path+"/"):
						found = true
						parent = slashpath.Join(pr[0], strings.TrimPrefix(parent, r.path))
					}
					newParents = append(newParents, parent)
				}
				if found {
					value.parents = newParents
				}
			}

			// mark naming conflict as resolved
			debugLog("marking naming conflict resolved for key: %s", r.key)
			opts.flattenContext.newRefs[r.key].isOAIGen = false
			opts.flattenContext.newRefs[r.key].resolved = true

			// determine if the previous substitution did inline a complex schema
			if r.schema != nil && r.schema.Ref.String() == "" { // inline schema
				asch, err := Schema(SchemaOpts{Schema: r.schema, Root: opts.Swagger(), BasePath: opts.BasePath})
				if err != nil {
					return false, err
				}
				debugLog("re-inlined schema: parent: %s, %t", pr[0], isAnalyzedAsComplex(asch))
				replacedWithComplex = replacedWithComplex ||
					!(slashpath.Dir(pr[0]) == definitionsPath) && isAnalyzedAsComplex(asch)
			}
		}
	}

	debugLog("replacedWithComplex: %t", replacedWithComplex)
	opts.Spec.reload() // re-analyze
	return replacedWithComplex, nil
}

// croak logs notifications and warnings about valid, but possibly unwanted constructs resulting
// from flattening a spec
func croak(opts *FlattenOpts) {
	reported := make(map[string]bool, len(opts.flattenContext.newRefs))
	for _, v := range opts.Spec.references.allRefs {
		// warns about duplicate handling
		for _, r := range opts.flattenContext.newRefs {
			if r.isOAIGen && r.path == v.String() {
				reported[r.newName] = true
			}
		}
	}
	for k := range reported {
		log.Printf("warning: duplicate flattened definition name resolved as %s", k)
	}
	// warns about possible type mismatches
	uniqueMsg := make(map[string]bool)
	for _, msg := range opts.flattenContext.warnings {
		if _, ok := uniqueMsg[msg]; ok {
			continue
		}
		log.Printf("warning: %s", msg)
		uniqueMsg[msg] = true
	}
}

// namePointers replaces all JSON pointers to anonymous documents by a $ref to a new named definitions.
//
// This is carried on depth-first. Pointers to $refs which are top level definitions are replaced by the $ref itself.
// Pointers to simple types are expanded, unless they express commonality (i.e. several such $ref are used).
func namePointers(opts *FlattenOpts) error {
	debugLog("name pointers")
	refsToReplace := make(map[string]SchemaRef, len(opts.Spec.references.schemas))
	for k, ref := range opts.Spec.references.allRefs {
		if slashpath.Dir(ref.String()) == definitionsPath {
			// this a ref to a top-level definition: ok
			continue
		}
		replacingRef, sch, erd := deepestRef(opts, ref)
		if erd != nil {
			return fmt.Errorf("at %s, %v", k, erd)
		}
		debugLog("planning pointer to replace at %s: %s, resolved to: %s", k, ref.String(), replacingRef.String())
		refsToReplace[k] = SchemaRef{
			Name:     k,            // caller
			Ref:      replacingRef, // callee
			Schema:   sch,
			TopLevel: slashpath.Dir(replacingRef.String()) == definitionsPath,
		}
	}
	depthFirst := sortDepthFirst(refsToReplace)
	namer := &inlineSchemaNamer{
		Spec:           opts.Swagger(),
		Operations:     opRefsByRef(gatherOperations(opts.Spec, nil)),
		flattenContext: opts.flattenContext,
		opts:           opts,
	}

	for _, key := range depthFirst {
		v := refsToReplace[key]
		// update current replacement, which may have been updated by previous changes of deeper elements
		replacingRef, sch, erd := deepestRef(opts, v.Ref)
		if erd != nil {
			return fmt.Errorf("at %s, %v", key, erd)
		}
		v.Ref = replacingRef
		v.Schema = sch
		v.TopLevel = slashpath.Dir(replacingRef.String()) == definitionsPath
		debugLog("replacing pointer at %s: resolved to: %s", key, v.Ref.String())

		if v.TopLevel {
			debugLog("replace pointer %s by canonical definition: %s", key, v.Ref.String())
			// if the schema is a $ref to a top level definition, just rewrite the pointer to this $ref
			if err := updateRef(opts.Swagger(), key, v.Ref); err != nil {
				return err
			}
		} else {
			// this is a JSON pointer to an anonymous document (internal or external):
			// create a definition for this schema when:
			// - it is a complex schema
			// - or it is pointed by more than one $ref (i.e. expresses commonality)
			// otherwise, expand the pointer (single reference to a simple type)
			//
			// The named definition for this follows the target's key, not the caller's
			debugLog("namePointers at %s for %s", key, v.Ref.String())

			// qualify the expanded schema
			/*
				if key == "#/paths/~1some~1where~1{id}/get/parameters/1/items" {
					// DEBUG
					//func getPointerFromKey(spec interface{}, key string) (string, interface{}, error) {
					k, res, err := getPointerFromKey(namer.Spec, key)
					debugLog("k = %s, res=%#v, err=%v", k, res, err)
				}
			*/
			asch, ers := Schema(SchemaOpts{Schema: v.Schema, Root: opts.Swagger(), BasePath: opts.BasePath})
			if ers != nil {
				return fmt.Errorf("schema analysis [%s]: %v", key, ers)
			}
			callers := make([]string, 0, 64)

			debugLog("looking for callers")
			an := New(opts.Swagger())
			for k, w := range an.references.allRefs {
				r, _, erd := deepestRef(opts, w)
				if erd != nil {
					return fmt.Errorf("at %s, %v", key, erd)
				}
				if r.String() == v.Ref.String() {
					callers = append(callers, k)
				}
			}
			debugLog("callers for %s: %d", v.Ref.String(), len(callers))
			if len(callers) == 0 {
				// has already been updated and resolved
				continue
			}

			parts := keyParts(v.Ref.String())
			debugLog("number of callers for %s: %d", v.Ref.String(), len(callers))
			// identifying edge case when the namer did nothing because we point to a non-schema object
			// no definition is created and we expand the $ref for all callers
			if (!asch.IsSimpleSchema || len(callers) > 1) && !parts.IsSharedParam() && !parts.IsSharedResponse() {
				debugLog("replace JSON pointer at [%s] by definition: %s", key, v.Ref.String())
				if err := namer.Name(v.Ref.String(), v.Schema, asch); err != nil {
					return err
				}

				// regular case: we named the $ref as a definition, and we move all callers to this new $ref
				for _, caller := range callers {
					if caller != key {
						// move $ref for next to resolve
						debugLog("identified caller of %s at [%s]", v.Ref.String(), caller)
						c := refsToReplace[caller]
						c.Ref = v.Ref
						refsToReplace[caller] = c
					}
				}
			} else {
				debugLog("expand JSON pointer for key=%s", key)
				if err := updateRefWithSchema(opts.Swagger(), key, v.Schema); err != nil {
					return err
				}
				// NOTE: there is no other caller to update
			}
		}
	}
	opts.Spec.reload() // re-analyze
	return nil
}

// deepestRef finds the first definition ref, from a cascade of nested refs which are not definitions.
//  - if no definition is found, returns the deepest ref.
//  - pointers to external files are expanded
//
// NOTE: all external $ref's are assumed to be already expanded at this stage.
func deepestRef(opts *FlattenOpts, ref swspec.Ref) (swspec.Ref, *swspec.Schema, error) {
	if !ref.HasFragmentOnly {
		// we found an external $ref, which is odd
		// does nothing on external $refs
		return ref, nil, nil
	}
	currentRef := ref
	visited := make(map[string]bool, 64)
DOWNREF:
	for currentRef.String() != "" {
		if slashpath.Dir(currentRef.String()) == definitionsPath {
			// this is a top-level definition: stop here and return this ref
			return currentRef, nil, nil
		}
		if _, beenThere := visited[currentRef.String()]; beenThere {
			return swspec.Ref{}, nil,
				fmt.Errorf("cannot resolve cyclic chain of pointers under %s", currentRef.String())
		}
		visited[currentRef.String()] = true
		value, _, err := currentRef.GetPointer().Get(opts.Swagger())
		if err != nil {
			return swspec.Ref{}, nil, err
		}
		switch refable := value.(type) {
		case *swspec.Schema:
			if refable.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = refable.Ref

		case swspec.Schema:
			if refable.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = refable.Ref

		case *swspec.SchemaOrArray:
			if refable.Schema == nil || refable.Schema != nil && refable.Schema.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = refable.Schema.Ref

		case *swspec.SchemaOrBool:
			if refable.Schema == nil || refable.Schema != nil && refable.Schema.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = refable.Schema.Ref

		case swspec.Response:
			// a pointer points to a schema initially marshalled in responses section...
			// Attempt to convert this to a schema. If this fails, the spec is invalid
			asJSON, _ := refable.MarshalJSON()
			var asSchema swspec.Schema
			err := asSchema.UnmarshalJSON(asJSON)
			if err != nil {
				return swspec.Ref{}, nil,
					fmt.Errorf("invalid type for resolved JSON pointer %s. Expected a schema a, got: %T",
						currentRef.String(), value)

			}
			opts.flattenContext.warnings = append(opts.flattenContext.warnings,
				fmt.Sprintf("found $ref %q (response) interpreted as schema", currentRef.String()))

			if asSchema.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = asSchema.Ref

		case swspec.Parameter:
			// a pointer points to a schema initially marshalled in parameters section...
			// Attempt to convert this to a schema. If this fails, the spec is invalid
			asJSON, _ := refable.MarshalJSON()
			var asSchema swspec.Schema
			err := asSchema.UnmarshalJSON(asJSON)
			if err != nil {
				return swspec.Ref{}, nil,
					fmt.Errorf("invalid type for resolved JSON pointer %s. Expected a schema a, got: %T",
						currentRef.String(), value)

			}
			opts.flattenContext.warnings = append(opts.flattenContext.warnings,
				fmt.Sprintf("found $ref %q (parameter) interpreted as schema", currentRef.String()))

			if asSchema.Ref.String() == "" {
				break DOWNREF
			}
			currentRef = asSchema.Ref

		default:
			return swspec.Ref{}, nil,
				fmt.Errorf("unhandled type to resolve JSON pointer %s. Expected a Schema, got: %T",
					currentRef.String(), value)

		}
	}
	// assess what schema we're ending with
	sch, erv := swspec.ResolveRefWithBase(opts.Swagger(), &currentRef, opts.ExpandOpts(false))
	if erv != nil {
		return swspec.Ref{}, nil, erv
	}
	if sch == nil {
		return swspec.Ref{}, nil, fmt.Errorf("no schema found at %s", currentRef.String())
	}
	return currentRef, sch, nil
}

// normalizeRef strips the current file from any $ref. This works around issue go-openapi/spec#76:
// leading absolute file in $ref is stripped
func normalizeRef(opts *FlattenOpts) error {
	debugLog("normalizeRef")
	opts.Spec.reload() // re-analyze
	for k, w := range opts.Spec.references.allRefs {
		if strings.HasPrefix(w.String(), opts.BasePath+definitionsPath) { // may be a mix of / and \, depending on OS
			// strip base path from definition
			debugLog("stripping absolute path for: %s", w.String())
			if err := updateRef(opts.Swagger(), k,
				swspec.MustCreateRef(slashpath.Join(definitionsPath, slashpath.Base(w.String())))); err != nil {
				return err
			}
		}
	}
	opts.Spec.reload() // re-analyze
	return nil
}
