package analysis

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"strconv"

	"github.com/go-openapi/jsonpointer"
	swspec "github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

// FlattenOpts configuration for flattening a swagger specification.
type FlattenOpts struct {
	// If Expand is true, we skip flattening the spec and expand it instead
	Expand   bool
	Spec     *Spec
	BasePath string

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

// Flatten an analyzed spec.
//
// To flatten a spec means:
//
// Expand the parameters, responses, path items, parameter items and header items.
// Import external (http, file) references so they become internal to the document.
// Move every inline schema to be a definition with an auto-generated name in a depth-first fashion.
// Rewritten schemas get a vendor extension x-go-gen-location so we know in which package they need to be rendered.
func Flatten(opts FlattenOpts) error {
	// Make sure opts.BasePath is an absolute path
	if !filepath.IsAbs(opts.BasePath) {
		cwd, _ := os.Getwd()
		opts.BasePath = filepath.Join(cwd, opts.BasePath)
	}
	// recursively expand responses, parameters, path items and items
	err := swspec.ExpandSpec(opts.Swagger(), &swspec.ExpandOptions{
		RelativeBase: opts.BasePath,
		SkipSchemas:  !opts.Expand,
	})
	if err != nil {
		return err
	}
	opts.Spec.reload() // re-analyze

	// at this point there are no other references left but schemas
	if err := importExternalReferences(&opts); err != nil {
		return err
	}
	opts.Spec.reload() // re-analyze

	// rewrite the inline schemas (schemas that aren't simple types or arrays of simple types)
	if err := nameInlinedSchemas(&opts); err != nil {
		return err
	}
	opts.Spec.reload() // re-analyze

	// TODO: simplifiy known schema patterns to flat objects with properties?
	return nil
}

func nameInlinedSchemas(opts *FlattenOpts) error {
	namer := &inlineSchemaNamer{Spec: opts.Swagger(), Operations: opRefsByRef(gatherOperations(opts.Spec, nil))}
	depthFirst := sortDepthFirst(opts.Spec.allSchemas)
	for _, key := range depthFirst {
		sch := opts.Spec.allSchemas[key]
		if sch.Schema != nil && sch.Schema.Ref.String() == "" && !sch.TopLevel { // inline schema
			asch, err := Schema(SchemaOpts{Schema: sch.Schema, Root: opts.Swagger(), BasePath: opts.BasePath})
			if err != nil {
				return fmt.Errorf("schema analysis [%s]: %v", sch.Ref.String(), err)
			}

			if !asch.IsSimpleSchema && !asch.IsArray { // complex schemas get moved
				if err := namer.Name(key, sch.Schema, asch); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

var depthGroupOrder = []string{"sharedOpParam", "opParam", "codeResponse", "defaultResponse", "definition"}

func sortDepthFirst(data map[string]SchemaRef) (sorted []string) {
	// group by category (shared params, op param, statuscode response, default response, definitions)
	// sort groups internally by number of parts in the key and lexical names
	// flatten groups into a single list of keys
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
		grouped[pk] = append(grouped[pk], key{len(split), k})
	}

	for _, pk := range depthGroupOrder {
		res := grouped[pk]
		sort.Sort(res)
		for _, v := range res {
			sorted = append(sorted, v.Key)
		}
	}

	return
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
	Spec       *swspec.Swagger
	Operations map[string]opRef
}

func opRefsByRef(oprefs map[string]opRef) map[string]opRef {
	result := make(map[string]opRef, len(oprefs))
	for _, v := range oprefs {
		result[v.Ref.String()] = v
	}
	return result
}

func (isn *inlineSchemaNamer) Name(key string, schema *swspec.Schema, aschema *AnalyzedSchema) error {
	if swspec.Debug {
		log.Printf("naming inlined schema at %s", key)
	}

	parts := keyParts(key)
	for _, name := range namesFromKey(parts, aschema, isn.Operations) {
		if name != "" {
			// create unique name
			newName := uniqifyName(isn.Spec.Definitions, swag.ToJSONName(name))

			// clone schema
			sch, err := cloneSchema(schema)
			if err != nil {
				return err
			}

			// replace values on schema
			if err := rewriteSchemaToRef(isn.Spec, key, swspec.MustCreateRef("#/definitions/"+newName)); err != nil {
				return fmt.Errorf("name inlined schema: %v", err)
			}

			sch.AddExtension("x-go-gen-location", genLocation(parts))
			// fmt.Printf("{\n  %q,\n  \"\",\n  spec.MustCreateRef(%q),\n  \"\",\n},\n", key, "#/definitions/"+newName)
			// save cloned schema to definitions
			saveSchema(isn.Spec, newName, sch)
		}
	}
	return nil
}

func genLocation(parts splitKey) string {
	if parts.IsOperation() {
		return "operations"
	}
	if parts.IsDefinition() {
		return "models"
	}
	return ""
}

func uniqifyName(definitions swspec.Definitions, name string) string {
	if name == "" {
		name = "oaiGen"
	}
	if len(definitions) == 0 {
		return name
	}

	unq := true
	for k := range definitions {
		if strings.ToLower(k) == strings.ToLower(name) {
			unq = false
			break
		}
	}

	if unq {
		return name
	}

	name += "OAIGen"
	var idx int
	unique := name
	_, known := definitions[unique]
	for known {
		idx++
		unique = fmt.Sprintf("%s%d", name, idx)
		_, known = definitions[unique]
	}
	return unique
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
	pths        = "paths"
	responses   = "responses"
	parameters  = "parameters"
	definitions = "definitions"
)

var ignoredKeys map[string]struct{}

func init() {
	ignoredKeys = map[string]struct{}{
		"schema":     {},
		"properties": {},
		"not":        {},
		"anyOf":      {},
		"oneOf":      {},
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

	if count%2 != 0 {
		return true
	}
	return false
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
	return len(s) > 1 && s[0] == pths
}

func (s splitKey) IsSharedOperationParam() bool {
	return len(s) > 2 && s[0] == pths && s[2] == parameters
}

func (s splitKey) IsOperationParam() bool {
	return len(s) > 3 && s[0] == pths && s[3] == parameters
}

func (s splitKey) IsOperationResponse() bool {
	return len(s) > 3 && s[0] == pths && s[3] == responses
}

func (s splitKey) IsDefaultResponse() bool {
	return len(s) > 4 && s[0] == pths && s[3] == responses && s[4] == "default"
}

func (s splitKey) IsStatusCodeResponse() bool {
	isInt := func() bool {
		_, err := strconv.Atoi(s[4])
		return err == nil
	}
	return len(s) > 4 && s[0] == pths && s[3] == responses && isInt()
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

var validMethods map[string]struct{}

func init() {
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

func (s splitKey) PathItemRef() swspec.Ref {
	if len(s) < 3 {
		return swspec.Ref{}
	}
	pth, method := s[1], s[2]
	if _, validMethod := validMethods[strings.ToUpper(method)]; !validMethod && !strings.HasPrefix(method, "x-") {
		return swspec.Ref{}
	}
	return swspec.MustCreateRef("#" + path.Join("/", pths, jsonpointer.Escape(pth), strings.ToUpper(method)))
}

func (s splitKey) PathRef() swspec.Ref {
	if !s.IsOperation() {
		return swspec.Ref{}
	}
	return swspec.MustCreateRef("#" + path.Join("/", pths, jsonpointer.Escape(s[1])))
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
	if swspec.Debug {
		log.Printf("rewriting schema to ref for %s with %s", key, ref.String())
	}
	pth := key[1:]
	ptr, err := jsonpointer.New(pth)
	if err != nil {
		return err
	}

	value, _, err := ptr.Get(spec)
	if err != nil {
		return err
	}

	switch refable := value.(type) {
	case *swspec.Schema:
		return rewriteParentRef(spec, key, ref)
	case *swspec.SchemaOrBool:
		if refable.Schema != nil {
			refable.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		}
	case *swspec.SchemaOrArray:
		if refable.Schema != nil {
			refable.Schema = &swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
		}
	case swspec.Schema:
		return rewriteParentRef(spec, key, ref)
	default:
		return fmt.Errorf("no schema with ref found at %s for %T", key, value)
	}

	return nil
}

func rewriteParentRef(spec *swspec.Swagger, key string, ref swspec.Ref) error {
	pth := key[1:]
	parent, entry := path.Dir(pth), path.Base(pth)
	if swspec.Debug {
		log.Println("getting schema holder at:", parent)
	}

	pptr, err := jsonpointer.New(parent)
	if err != nil {
		return err
	}
	pvalue, _, err := pptr.Get(spec)
	if err != nil {
		return fmt.Errorf("can't get parent for %s: %v", parent, err)
	}
	if swspec.Debug {
		log.Printf("rewriting holder for %T", pvalue)
	}

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
			return fmt.Errorf("%s not a number: %v", pth, err)
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
			return fmt.Errorf("%s not a number: %v", pth, err)
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
			return fmt.Errorf("%s not a number: %v", pth, err)
		}
		container[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

	case *swspec.SchemaOrArray:
		idx, err := strconv.Atoi(entry)
		if err != nil {
			return fmt.Errorf("%s not a number: %v", pth, err)
		}
		container.Schemas[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}
	default:
		return fmt.Errorf("unhandled parent schema rewrite %s (%T)", key, pvalue)
	}
	return nil
}

func cloneSchema(schema *swspec.Schema) (*swspec.Schema, error) {
	var sch swspec.Schema
	if err := swag.FromDynamicJSON(schema, &sch); err != nil {
		return nil, fmt.Errorf("name inlined schema: %v", err)
	}
	return &sch, nil
}

func importExternalReferences(opts *FlattenOpts) error {
	groupedRefs := reverseIndexForSchemaRefs(opts)

	for refStr, entry := range groupedRefs {
		if !entry.Ref.HasFragmentOnly {
			if swspec.Debug {
				log.Printf("importing external schema for [%s] from %s", strings.Join(entry.Keys, ", "), refStr)
			}
			// resolve to actual schema
			sch := new(swspec.Schema)
			sch.Ref = entry.Ref
			expandOpts := swspec.ExpandOptions{
				RelativeBase: opts.BasePath,
				SkipSchemas:  false,
			}
			err := swspec.ExpandSchemaWithBasePath(sch, nil, &expandOpts)
			if err != nil {
				return err
			}
			if sch == nil {
				return fmt.Errorf("no schema found at %s for [%s]", refStr, strings.Join(entry.Keys, ", "))
			}
			if swspec.Debug {
				log.Printf("importing external schema for [%s] from %s", strings.Join(entry.Keys, ", "), refStr)
			}

			// generate a unique name
			newName := uniqifyName(opts.Swagger().Definitions, nameFromRef(entry.Ref))
			if swspec.Debug {
				log.Printf("new name for [%s]: %s", strings.Join(entry.Keys, ", "), newName)
			}

			// rewrite the external refs to local ones
			for _, key := range entry.Keys {
				if err := updateRef(opts.Swagger(), key, swspec.MustCreateRef("#"+path.Join("/definitions", newName))); err != nil {
					return err
				}
			}

			// add the resolved schema to the definitions
			saveSchema(opts.Swagger(), newName, sch)
		}
	}
	return nil
}

type refRevIdx struct {
	Ref  swspec.Ref
	Keys []string
}

func reverseIndexForSchemaRefs(opts *FlattenOpts) map[string]refRevIdx {
	collected := make(map[string]refRevIdx)
	for key, schRef := range opts.Spec.references.schemas {
		if entry, ok := collected[schRef.String()]; ok {
			entry.Keys = append(entry.Keys, key)
			collected[schRef.String()] = entry
		} else {
			collected[schRef.String()] = refRevIdx{
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
		return swag.ToJSONName(path.Base(u.Fragment))
	}
	if u.Path != "" {
		bn := path.Base(u.Path)
		if bn != "" && bn != "/" {
			ext := path.Ext(bn)
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

func updateRef(spec *swspec.Swagger, key string, ref swspec.Ref) error {
	if swspec.Debug {
		log.Printf("updating ref for %s with %s", key, ref.String())
	}
	pth := key[1:]
	ptr, err := jsonpointer.New(pth)
	if err != nil {
		return err
	}

	value, _, err := ptr.Get(spec)
	if err != nil {
		return err
	}

	switch refable := value.(type) {
	case *swspec.Schema:
		refable.Ref = ref
	case *swspec.SchemaOrBool:
		if refable.Schema != nil {
			refable.Schema.Ref = ref
		}
	case *swspec.SchemaOrArray:
		if refable.Schema != nil {
			refable.Schema.Ref = ref
		}
	case swspec.Schema:
		parent, entry := path.Dir(pth), path.Base(pth)
		if swspec.Debug {
			log.Println("getting schema holder at:", parent)
		}

		pptr, err := jsonpointer.New(parent)
		if err != nil {
			return err
		}
		pvalue, _, err := pptr.Get(spec)
		if err != nil {
			return fmt.Errorf("can't get parent for %s: %v", parent, err)
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
			idx, err := strconv.Atoi(entry)
			if err != nil {
				return fmt.Errorf("%s not a number: %v", pth, err)
			}
			container.Schemas[idx] = swspec.Schema{SchemaProps: swspec.SchemaProps{Ref: ref}}

		}

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
				Ref:    swspec.MustCreateRef("#" + path.Join("/paths", jsonpointer.Escape(pth), method)),
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
