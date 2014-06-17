// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// goGenVersion is the version of the Go code generator
const goGenVersion = "0.5"

var (
	apiToGenerate = flag.String("api", "*", "The API ID to generate, like 'tasks:v1'. A value of '*' means all.")
	useCache      = flag.Bool("cache", true, "Use cache of discovered Google API discovery documents.")
	genDir        = flag.String("gendir", "", "Directory to use to write out generated Go files")
	build         = flag.Bool("build", false, "Compile generated packages.")
	install       = flag.Bool("install", false, "Install generated packages.")
	apisURL       = flag.String("discoveryurl", "https://www.googleapis.com/discovery/v1/apis", "URL to root discovery document")

	publicOnly = flag.Bool("publiconly", true, "Only build public, released APIs. Only applicable for Google employees.")

	jsonFile     = flag.String("api_json_file", "", "If non-empty, the path to a local file on disk containing the API to generate. Exclusive with setting --api.")
	output       = flag.String("output", "", "(optional) Path to source output file. If not specified, the API name and version are used to construct an output path (e.g. tasks/v1).")
	googleAPIPkg = flag.String("googleapi_pkg", "code.google.com/p/google-api-go-client/googleapi", "Go package path of the 'googleapi' support package.")
)

// API represents an API to generate, as well as its state while it's
// generating.
type API struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Version       string `json:"version"`
	Title         string `json:"title"`
	DiscoveryLink string `json:"discoveryLink"` // relative
	RootURL       string `json:"rootUrl"`
	ServicePath   string `json:"servicePath"`
	Preferred     bool   `json:"preferred"`

	m map[string]interface{}

	forceJSON []byte // if non-nil, the JSON schema file. else fetched.
	usedNames namePool
	schemas   map[string]*Schema // apiName -> schema

	p  func(format string, args ...interface{}) // print raw
	pn func(format string, args ...interface{}) // print with indent and newline
}

func (a *API) sortedSchemaNames() (names []string) {
	for name := range a.schemas {
		names = append(names, name)
	}
	sort.Strings(names)
	return
}

type AllAPIs struct {
	Items []*API `json:"items"`
}

type generateError struct {
	api   *API
	error error
}

func (e *generateError) Error() string {
	return fmt.Sprintf("API %s failed to generate code: %v", e.api.ID, e.error)
}

type compileError struct {
	api    *API
	output string
}

func (e *compileError) Error() string {
	return fmt.Sprintf("API %s failed to compile:\n%v", e.api.ID, e.output)
}

func main() {
	flag.Parse()

	if *install {
		*build = true
	}

	var (
		apiIds  = []string{}
		matches = []*API{}
		errors  = []error{}
	)
	for _, api := range getAPIs() {
		apiIds = append(apiIds, api.ID)
		if !api.want() {
			continue
		}
		matches = append(matches, api)
		log.Printf("Generating API %s", api.ID)
		err := api.WriteGeneratedCode()
		if err != nil {
			errors = append(errors, &generateError{api, err})
			continue
		}
		if *build {
			var args []string
			if *install {
				args = append(args, "install")
			} else {
				args = append(args, "build")
			}
			args = append(args, api.Target())
			out, err := exec.Command("go", args...).CombinedOutput()
			if err != nil {
				errors = append(errors, &compileError{api, string(out)})
			}
		}
	}

	if len(matches) == 0 {
		log.Fatalf("No APIs matched %q; options are %v", *apiToGenerate, apiIds)
	}

	if len(errors) > 0 {
		log.Printf("%d API(s) failed to generate or compile:", len(errors))
		for _, ce := range errors {
			log.Printf(ce.Error())
		}
		os.Exit(1)
	}
}

func (a *API) want() bool {
	if strings.Contains(a.ID, "buzz") {
		// R.I.P.
		return false
	}
	if strings.Contains(a.ID, "fusiontables") {
		// TODO(bradfitz): broken codegen.
		return false
	}
	return *apiToGenerate == "*" || *apiToGenerate == a.ID
}

func getAPIs() []*API {
	if *jsonFile != "" {
		return getAPIsFromFile()
	}
	var all AllAPIs
	disco := slurpURL(*apisURL)
	if err := json.Unmarshal(disco, &all); err != nil {
		log.Fatalf("error decoding JSON in %s: %v", apisURL, err)
	}
	if !*publicOnly && *apiToGenerate != "*" {
		parts := strings.SplitN(*apiToGenerate, ":", 2)
		apiName := parts[0]
		apiVersion := parts[1]
		all.Items = append(all.Items, &API{
			ID:            *apiToGenerate,
			Name:          apiName,
			Version:       apiVersion,
			DiscoveryLink: fmt.Sprintf("./apis/%s/%s/rest", apiName, apiVersion),
		})
	}
	return all.Items
}

// getAPIsFromFile handles the case of generating exactly one API
// from the flag given in --api_json_file
func getAPIsFromFile() []*API {
	if *apiToGenerate != "*" {
		log.Fatalf("Can't set --api with --api_json_file.")
	}
	if !*publicOnly {
		log.Fatalf("Can't set --publiconly with --api_json_file.")
	}
	a, err := apiFromFile(*jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	return []*API{a}
}

func apiFromFile(file string) (*API, error) {
	jsonBytes, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("Error reading %s: %v", file, err)
	}
	a := &API{
		forceJSON: jsonBytes,
	}
	if err := json.Unmarshal(jsonBytes, a); err != nil {
		return nil, fmt.Errorf("Decoding JSON in %s: %v", file, err)
	}
	return a, nil
}

func writeFile(file string, contents []byte) error {
	// Don't write it if the contents are identical.
	existing, err := ioutil.ReadFile(file)
	if err == nil && bytes.Equal(existing, contents) {
		return nil
	}
	return ioutil.WriteFile(file, contents, 0644)
}

func slurpURL(urlStr string) []byte {
	diskFile := filepath.Join(os.TempDir(), "google-api-cache-"+url.QueryEscape(urlStr))
	if *useCache {
		bs, err := ioutil.ReadFile(diskFile)
		if err == nil && len(bs) > 0 {
			return bs
		}
	}

	req, err := http.NewRequest("GET", urlStr, nil)
	if err != nil {
		log.Fatal(err)
	}
	if *publicOnly {
		req.Header.Add("X-User-IP", "0.0.0.0") // hack
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatalf("Error fetching URL %s: %v", urlStr, err)
	}
	bs, err := ioutil.ReadAll(res.Body)
	if err != nil {
		log.Fatalf("Error reading body of URL %s: %v", urlStr, err)
	}
	if *useCache {
		if err := ioutil.WriteFile(diskFile, bs, 0666); err != nil {
			log.Printf("Warning: failed to write JSON of %s to disk file %s: %v", urlStr, diskFile, err)
		}
	}
	return bs
}

func panicf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
}

// namePool keeps track of used names and assigns free ones based on a
// preferred name
type namePool struct {
	m map[string]bool // lazily initialized
}

func (p *namePool) Get(preferred string) string {
	if p.m == nil {
		p.m = make(map[string]bool)
	}
	name := preferred
	tries := 0
	for p.m[name] {
		tries++
		name = fmt.Sprintf("%s%d", preferred, tries)
	}
	p.m[name] = true
	return name
}

func (a *API) SourceDir() string {
	if *genDir == "" {
		paths := filepath.SplitList(os.Getenv("GOPATH"))
		if len(paths) > 0 && paths[0] != "" {
			*genDir = filepath.Join(paths[0], "src", "code.google.com", "p", "google-api-go-client")
		}
	}
	return filepath.Join(*genDir, a.Package(), a.Version)
}

func (a *API) DiscoveryURL() string {
	if a.DiscoveryLink == "" {
		log.Fatalf("API %s has no DiscoveryLink", a.ID)
	}
	base, _ := url.Parse(*apisURL)
	u, err := base.Parse(a.DiscoveryLink)
	if err != nil {
		log.Fatalf("API %s has bogus DiscoveryLink %s: %v", a.ID, a.DiscoveryLink, err)
	}
	return u.String()
}

func (a *API) Package() string {
	return strings.ToLower(a.Name)
}

func (a *API) Target() string {
	return fmt.Sprintf("code.google.com/p/google-api-go-client/%s/%s", a.Package(), a.Version)
}

// GetName returns a free top-level function/type identifier in the package.
// It tries to return your preferred match if it's free.
func (a *API) GetName(preferred string) string {
	return a.usedNames.Get(preferred)
}

func (a *API) apiBaseURL() string {
	if a.RootURL != "" {
		return a.RootURL + a.ServicePath
	}
	return resolveRelative(*apisURL, jstr(a.m, "basePath"))
}

func (a *API) needsDataWrapper() bool {
	for _, feature := range jstrlist(a.m, "features") {
		if feature == "dataWrapper" {
			return true
		}
	}
	return false
}

func (a *API) jsonBytes() []byte {
	if v := a.forceJSON; v != nil {
		return v
	}
	return slurpURL(a.DiscoveryURL())
}

func (a *API) WriteGeneratedCode() error {
	outdir := a.SourceDir()
	err := os.MkdirAll(outdir, 0755)
	if err != nil {
		return fmt.Errorf("failed to Mkdir %s: %v", outdir, err)
	}

	pkg := a.Package()
	writeFile(filepath.Join(outdir, a.Package()+"-api.json"), a.jsonBytes())

	genfilename := *output
	if genfilename == "" {
		genfilename = filepath.Join(outdir, pkg+"-gen.go")
	}

	code, err := a.GenerateCode()
	errw := writeFile(genfilename, code)
	if err == nil {
		err = errw
	}
	return err
}

func (a *API) GenerateCode() ([]byte, error) {
	pkg := a.Package()

	a.m = make(map[string]interface{})
	m := a.m
	jsonBytes := a.jsonBytes()
	err := json.Unmarshal(jsonBytes, &a.m)
	if err != nil {
		return nil, err
	}

	// Buffer the output in memory, for gofmt'ing later in the defer.
	var buf bytes.Buffer
	a.p = func(format string, args ...interface{}) {
		_, err := fmt.Fprintf(&buf, format, args...)
		if err != nil {
			panic(err)
		}
	}
	a.pn = func(format string, args ...interface{}) {
		a.p(format+"\n", args...)
	}

	p, pn := a.p, a.pn
	reslist := a.Resources(a.m, "")

	p("// Package %s provides access to the %s.\n", pkg, jstr(m, "title"))
	if docs := jstr(m, "documentationLink"); docs != "" {
		p("//\n")
		p("// See %s\n", docs)
	}
	p("//\n// Usage example:\n")
	p("//\n")
	p("//   import %q\n", a.Target())
	p("//   ...\n")
	p("//   %sService, err := %s.New(oauthHttpClient)\n", pkg, pkg)

	p("package %s\n", pkg)
	p("\n")
	p("import (\n")
	for _, pkg := range []string{
		"bytes",
		*googleAPIPkg,
		"encoding/json",
		"errors",
		"fmt",
		"io",
		"net/http",
		"net/url",
		"strconv",
		"strings",
	} {
		p("\t%q\n", pkg)
	}
	p(")\n\n")
	pn("// Always reference these packages, just in case the auto-generated code")
	pn("// below doesn't.")
	pn("var _ = bytes.NewBuffer")
	pn("var _ = strconv.Itoa")
	pn("var _ = fmt.Sprintf")
	pn("var _ = json.NewDecoder")
	pn("var _ = io.Copy")
	pn("var _ = url.Parse")
	pn("var _ = googleapi.Version")
	pn("var _ = errors.New")
	pn("var _ = strings.Replace")
	pn("")
	pn("const apiId = %q", jstr(m, "id"))
	pn("const apiName = %q", jstr(m, "name"))
	pn("const apiVersion = %q", jstr(m, "version"))
	p("const basePath = %q\n", a.apiBaseURL())
	p("\n")

	a.generateScopeConstants()

	a.GetName("New") // ignore return value; we're the first caller
	pn("func New(client *http.Client) (*Service, error) {")
	pn("if client == nil { return nil, errors.New(\"client is nil\") }")
	pn("s := &Service{client: client, BasePath: basePath}")
	for _, res := range reslist { // add top level resources.
		pn("s.%s = New%s(s)", res.GoField(), res.GoType())
	}
	pn("return s, nil")
	pn("}")

	a.GetName("Service") // ignore return value; no user-defined names yet
	p("\ntype Service struct {\n")
	p("\tclient *http.Client\n")
	p("\tBasePath string // API endpoint base URL\n")

	for _, res := range reslist {
		p("\n\t%s\t*%s\n", res.GoField(), res.GoType())
	}
	p("}\n")

	for _, res := range reslist {
		res.generateType()
	}

	a.PopulateSchemas()

	for _, name := range a.sortedSchemaNames() {
		a.schemas[name].writeSchemaCode()
	}

	for _, meth := range a.APIMethods() {
		meth.generateCode()
	}

	for _, res := range reslist {
		res.generateMethods()
	}

	clean, err := format.Source(buf.Bytes())
	if err != nil {
		return buf.Bytes(), err
	}
	return clean, nil
}

func (a *API) generateScopeConstants() {
	auth := jobj(a.m, "auth")
	if auth == nil {
		return
	}
	oauth2 := jobj(auth, "oauth2")
	if oauth2 == nil {
		return
	}
	scopes := jobj(oauth2, "scopes")
	if scopes == nil || len(scopes) == 0 {
		return
	}

	a.p("// OAuth2 scopes used by this API.\n")
	a.p("const (\n")
	n := 0
	for _, scopeName := range sortedKeys(scopes) {
		mi := scopes[scopeName]
		if n > 0 {
			a.p("\n")
		}
		n++
		ident := scopeIdentifierFromURL(scopeName)
		if des := jstr(mi.(map[string]interface{}), "description"); des != "" {
			a.p("%s", asComment("\t", des))
		}
		a.p("\t%s = %q\n", ident, scopeName)
	}
	a.p(")\n\n")
}

func scopeIdentifierFromURL(urlStr string) string {
	const prefix = "https://www.googleapis.com/auth/"
	if !strings.HasPrefix(urlStr, prefix) {
		log.Fatalf("Unexpected oauth2 scope %q doesn't start with %q", urlStr, prefix)
	}
	ident := validGoIdentifer(initialCap(urlStr[len(prefix):])) + "Scope"
	return ident
}

type Schema struct {
	api *API
	m   map[string]interface{} // original JSON map

	typ *Type // lazily populated by Type

	apiName string // the native API-defined name of this type
	goName  string // lazily populated by GoName
}

type Property struct {
	s       *Schema                // property of which schema
	apiName string                 // the native API-defined name of this property
	m       map[string]interface{} // original JSON map

	typ *Type // lazily populated by Type
}

func (p *Property) Type() *Type {
	if p.typ == nil {
		p.typ = &Type{api: p.s.api, m: p.m}
	}
	return p.typ
}

func (p *Property) GoName() string {
	return initialCap(p.apiName)
}

func (p *Property) APIName() string {
	return p.apiName
}

func (p *Property) Description() string {
	return jstr(p.m, "description")
}

type Type struct {
	m   map[string]interface{} // JSON map containing key "type" and maybe "items", "properties"
	api *API
}

func (t *Type) apiType() string {
	// Note: returns "" on reference types
	if t, ok := t.m["type"].(string); ok {
		return t
	}
	return ""
}

func (t *Type) apiTypeFormat() string {
	if f, ok := t.m["format"].(string); ok {
		return f
	}
	return ""
}

func (t *Type) isIntAsString() bool {
	return t.apiType() == "string" && strings.Contains(t.apiTypeFormat(), "int")
}

func (t *Type) asSimpleGoType() (goType string, ok bool) {
	return simpleTypeConvert(t.apiType(), t.apiTypeFormat())
}

func (t *Type) String() string {
	return fmt.Sprintf("[type=%q, map=%s]", t.apiType(), prettyJSON(t.m))
}

func (t *Type) AsGo() string {
	if t, ok := t.asSimpleGoType(); ok {
		return t
	}
	if at, ok := t.ArrayType(); ok {
		if at.apiType() == "string" {
			switch at.apiTypeFormat() {
			case "int64":
				return "googleapi.Int64s"
			case "uint64":
				return "googleapi.Uint64s"
			case "int32":
				return "googleapi.Int32s"
			case "uint32":
				return "googleapi.Uint32s"
			case "float64":
				return "googleapi.Float64s"
			default:
				return "[]" + at.AsGo()
			}
		}
		return "[]" + at.AsGo()
	}
	if ref, ok := t.Reference(); ok {
		s := t.api.schemas[ref]
		if s == nil {
			panic(fmt.Sprintf("in Type.AsGo(), failed to find referenced type %q for %s",
				ref, prettyJSON(t.m)))
		}
		return s.Type().AsGo()
	}
	if t.IsMap() {
		// TODO(gmlewis): support maps to any type.
		return fmt.Sprintf("map[string]string")
	}
	if t.IsStruct() {
		if apiName, ok := t.m["_apiName"].(string); ok {
			s := t.api.schemas[apiName]
			if s == nil {
				panic(fmt.Sprintf("in Type.AsGo, _apiName of %q didn't point to a valid schema; json: %s",
					apiName, prettyJSON(t.m)))
			}
			return "*" + s.GoName()
		}
		panic("in Type.AsGo, no _apiName found for struct type " + prettyJSON(t.m))
	}
	panic("unhandled Type.AsGo for " + prettyJSON(t.m))
}

func (t *Type) IsSimple() bool {
	_, ok := simpleTypeConvert(t.apiType(), t.apiTypeFormat())
	return ok
}

func (t *Type) IsStruct() bool {
	return t.apiType() == "object"
}

func (t *Type) Reference() (apiName string, ok bool) {
	apiName = jstr(t.m, "$ref")
	ok = apiName != ""
	return
}

func (t *Type) IsMap() bool {
	props := jobj(t.m, "additionalProperties")
	if props == nil {
		return false
	}
	s := jstr(props, "type")
	b := s == "string"
	if !b {
		log.Printf("Warning: found map to type %q which is not implemented yet.", s)
	}
	return b
}

func (t *Type) IsReference() bool {
	return jstr(t.m, "$ref") != ""
}

func (t *Type) ReferenceSchema() (s *Schema, ok bool) {
	apiName, ok := t.Reference()
	if !ok {
		return
	}

	s = t.api.schemas[apiName]
	if s == nil {
		panicf("failed to find t.api.schemas[%q] while resolving reference",
			apiName)
	}
	return s, true
}

func (t *Type) ArrayType() (elementType *Type, ok bool) {
	if t.apiType() != "array" {
		return
	}
	items := jobj(t.m, "items")
	if items == nil {
		panicf("can't handle array type missing its 'items' key. map is %#v", t.m)
	}
	return &Type{api: t.api, m: items}, true
}

func (s *Schema) Type() *Type {
	if s.typ == nil {
		s.typ = &Type{api: s.api, m: s.m}
	}
	return s.typ
}

func (s *Schema) properties() []*Property {
	if !s.Type().IsStruct() {
		panic("called properties on non-object schema")
	}
	pl := []*Property{}
	propMap := jobj(s.m, "properties")
	for _, name := range sortedKeys(propMap) {
		m := propMap[name].(map[string]interface{})
		pl = append(pl, &Property{
			s:       s,
			m:       m,
			apiName: name,
		})
	}
	return pl
}

func (s *Schema) populateSubSchemas() (outerr error) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		outerr = fmt.Errorf("%v", r)
	}()

	addSubStruct := func(subApiName string, t *Type) {
		if s.api.schemas[subApiName] != nil {
			panic("dup schema apiName: " + subApiName)
		}
		subm := t.m
		subm["_apiName"] = subApiName
		subs := &Schema{
			api:     s.api,
			m:       subm,
			typ:     t,
			apiName: subApiName,
		}
		s.api.schemas[subApiName] = subs
		err := subs.populateSubSchemas()
		if err != nil {
			panicf("in sub-struct %q: %v", subApiName, err)
		}
	}

	if s.Type().IsStruct() {
		for _, p := range s.properties() {
			if p.Type().IsSimple() || p.Type().IsMap() {
				continue
			}
			if at, ok := p.Type().ArrayType(); ok {
				if at.IsSimple() || at.IsReference() {
					continue
				}
				subApiName := fmt.Sprintf("%s.%s", s.apiName, p.apiName)
				if at.IsStruct() {
					addSubStruct(subApiName, at) // was p.Type()?
					continue
				}
				if _, ok := at.ArrayType(); ok {
					addSubStruct(subApiName, at)
					continue
				}
				panicf("Unknown property array type for %q: %s", subApiName, at)
				continue
			}
			subApiName := fmt.Sprintf("%s.%s", s.apiName, p.apiName)
			if p.Type().IsStruct() {
				addSubStruct(subApiName, p.Type())
				continue
			}
			if p.Type().IsReference() {
				continue
			}
			panicf("Unknown type for %q: %s", subApiName, p.Type())
		}
		return
	}

	if at, ok := s.Type().ArrayType(); ok {
		if at.IsSimple() || at.IsReference() {
			return
		}
		subApiName := fmt.Sprintf("%s.Item", s.apiName)

		if at.IsStruct() {
			addSubStruct(subApiName, at)
			return
		}
		if at, ok := at.ArrayType(); ok {
			if at.IsSimple() || at.IsReference() {
				return
			}
			addSubStruct(subApiName, at)
			return
		}
		panicf("Unknown array type for %q: %s", subApiName, at)
		return
	}

	if s.Type().IsSimple() || s.Type().IsReference() {
		return
	}

	fmt.Fprintf(os.Stderr, "in populateSubSchemas, schema is: %s", prettyJSON(s.m))
	panicf("populateSubSchemas: unsupported type for schema %q", s.apiName)
	panic("unreachable")
}

// GoName returns (or creates and returns) the bare Go name
// of the apiName, making sure that it's a proper Go identifier
// and doesn't conflict with an existing name.
func (s *Schema) GoName() string {
	if s.goName == "" {
		s.goName = s.api.GetName(initialCap(s.apiName))
	}
	return s.goName
}

func (s *Schema) writeSchemaCode() {
	if s.Type().IsStruct() && !s.Type().IsMap() {
		s.writeSchemaStruct()
		return
	}

	if _, ok := s.Type().ArrayType(); ok {
		log.Printf("TODO writeSchemaCode for arrays for %s", s.GoName())
		return
	}

	if destSchema, ok := s.Type().ReferenceSchema(); ok {
		// Convert it to a struct using embedding.
		s.api.p("\ntype %s struct {\n", s.GoName())
		s.api.p("\t%s\n", destSchema.GoName())
		s.api.p("}\n")
		return
	}

	if s.Type().IsSimple() || s.Type().IsMap() {
		return
	}

	fmt.Fprintf(os.Stderr, "in writeSchemaCode, schema is: %s", prettyJSON(s.m))
	panicf("writeSchemaCode: unsupported type for schema %q", s.apiName)
}

func (s *Schema) writeSchemaStruct() {
	// TODO: description
	s.api.p("\ntype %s struct {\n", s.GoName())
	for i, p := range s.properties() {
		if i > 0 {
			s.api.p("\n")
		}
		pname := p.GoName()
		if des := p.Description(); des != "" {
			s.api.p("%s", asComment("\t", fmt.Sprintf("%s: %s", pname, des)))
		}
		var extraOpt string
		if p.Type().isIntAsString() {
			extraOpt += ",string"
		}
		s.api.p("\t%s %s `json:\"%s,omitempty%s\"`\n", pname, p.Type().AsGo(), p.APIName(), extraOpt)
	}
	s.api.p("}\n")
}

// PopulateSchemas reads all the API types ("schemas") from the JSON file
// and converts them to *Schema instances, returning an identically
// keyed map, additionally containing subresources.  For instance,
//
// A resource "Foo" of type "object" with a property "bar", also of type
// "object" (an anonymous sub-resource), will get a synthetic API name
// of "Foo.bar".
//
// A resource "Foo" of type "array" with an "items" of type "object"
// will get a synthetic API name of "Foo.Item".
func (a *API) PopulateSchemas() {
	m := jobj(a.m, "schemas")
	if a.schemas != nil {
		panic("")
	}
	a.schemas = make(map[string]*Schema)
	for name, mi := range m {
		s := &Schema{
			api:     a,
			apiName: name,
			m:       mi.(map[string]interface{}),
		}

		// And a little gross hack, so a map alone is good
		// enough to get its apiName:
		s.m["_apiName"] = name

		a.schemas[name] = s
		err := s.populateSubSchemas()
		if err != nil {
			panicf("Error populating schema with API name %q: %v", name, err)
		}
	}
}

type Resource struct {
	api       *API
	name      string
	parent    string
	m         map[string]interface{}
	resources []*Resource
}

func (r *Resource) generateType() {
	p, pn := r.api.p, r.api.pn
	t := r.GoType()
	pn(fmt.Sprintf("func New%s(s *Service) *%s {", t, t))
	pn("rs := &%s{s : s}", t)
	for _, res := range r.resources {
		pn("rs.%s = New%s(s)", res.GoField(), res.GoType())
	}
	pn("return rs")
	pn("}")

	p("\ntype %s struct {\n", t)
	p("\ts *Service\n")
	for _, res := range r.resources {
		p("\n\t%s\t*%s\n", res.GoField(), res.GoType())
	}
	p("}\n")

	for _, res := range r.resources {
		res.generateType()
	}
}

func (r *Resource) generateMethods() {
	for _, meth := range r.Methods() {
		meth.generateCode()
	}
	for _, res := range r.resources {
		res.generateMethods()
	}
}

func (r *Resource) GoField() string {
	return initialCap(r.name)
}

func (r *Resource) GoType() string {
	return initialCap(fmt.Sprintf("%s.%s", r.parent, r.name)) + "Service"
}

func (r *Resource) Methods() []*Method {
	ms := []*Method{}

	methMap := jobj(r.m, "methods")
	for _, mname := range sortedKeys(methMap) {
		mi := methMap[mname]
		ms = append(ms, &Method{
			api:  r.api,
			r:    r,
			name: mname,
			m:    mi.(map[string]interface{}),
		})
	}
	return ms
}

type Method struct {
	api  *API
	r    *Resource // or nil if a API-level (top-level) method
	name string
	m    map[string]interface{} // original JSON

	params []*Param // all Params, of each type, lazily set by first access to Parameters
}

func (m *Method) Id() string {
	return jstr(m.m, "id")
}

func (m *Method) supportsMedia() bool {
	return jobj(m.m, "mediaUpload") != nil
}

func (m *Method) mediaPath() string {
	return jstr(jobj(jobj(jobj(m.m, "mediaUpload"), "protocols"), "simple"), "path")
}

func (m *Method) Params() []*Param {
	if m.params == nil {
		paramMap := jobj(m.m, "parameters")
		for _, name := range sortedKeys(paramMap) {
			mi := paramMap[name]
			pm := mi.(map[string]interface{})
			m.params = append(m.params, &Param{
				name:   name,
				m:      pm,
				method: m,
			})
		}
	}
	return m.params
}

func (m *Method) grepParams(f func(*Param) bool) []*Param {
	matches := make([]*Param, 0)
	for _, param := range m.Params() {
		if f(param) {
			matches = append(matches, param)
		}
	}
	return matches
}

func (m *Method) NamedParam(name string) *Param {
	matches := m.grepParams(func(p *Param) bool {
		return p.name == name
	})
	if len(matches) < 1 {
		log.Panicf("failed to find named parameter %q", name)
	}
	if len(matches) > 1 {
		log.Panicf("found multiple parameters for parameter name %q", name)
	}
	return matches[0]
}

func (m *Method) OptParams() []*Param {
	return m.grepParams(func(p *Param) bool {
		return !p.IsRequired()
	})
}

func (m *Method) RequiredRepeatedQueryParams() []*Param {
	return m.grepParams(func(p *Param) bool {
		return p.IsRequired() && p.IsRepeated() && p.Location() == "query"
	})
}

func (m *Method) RequiredQueryParams() []*Param {
	return m.grepParams(func(p *Param) bool {
		return p.IsRequired() && !p.IsRepeated() && p.Location() == "query"
	})
}

func (meth *Method) generateCode() {
	res := meth.r // may be nil if a top-level method
	a := meth.api
	p, pn := a.p, a.pn

	pn("\n// method id %q:", meth.Id())

	retTypeComma := responseType(a, meth.m)
	if retTypeComma != "" {
		retTypeComma += ", "
	}

	args := meth.NewArguments()
	methodName := initialCap(meth.name)

	prefix := ""
	if res != nil {
		prefix = initialCap(fmt.Sprintf("%s.%s", res.parent, res.name))
	}
	callName := a.GetName(prefix + methodName + "Call")

	p("\ntype %s struct {\n", callName)
	p("\ts *Service\n")
	for _, arg := range args.l {
		p("\t%s %s\n", arg.goname, arg.gotype)
	}
	p("\topt_ map[string]interface{}\n")
	if meth.supportsMedia() {
		p("\tmedia_ io.Reader\n")
	}
	p("}\n")

	p("\n%s", asComment("", methodName+": "+jstr(meth.m, "description")))

	var servicePtr string
	if res == nil {
		p("func (s *Service) %s(%s) *%s {\n", methodName, args, callName)
		servicePtr = "s"
	} else {
		p("func (r *%s) %s(%s) *%s {\n", res.GoType(), methodName, args, callName)
		servicePtr = "r.s"
	}

	p("\tc := &%s{s: %s, opt_: make(map[string]interface{})}\n", callName, servicePtr)
	for _, arg := range args.l {
		p("\tc.%s = %s\n", arg.goname, arg.goname)
	}
	p("\treturn c\n")
	p("}\n")

	for _, opt := range meth.OptParams() {
		setter := initialCap(opt.name)
		des := jstr(opt.m, "description")
		des = strings.Replace(des, "Optional.", "", 1)
		des = strings.TrimSpace(des)
		p("\n%s", asComment("", fmt.Sprintf("%s sets the optional parameter %q: %s", setter, opt.name, des)))
		np := new(namePool)
		np.Get("c") // take the receiver's name
		paramName := np.Get(validGoIdentifer(opt.name))
		p("func (c *%s) %s(%s %s) *%s {\n", callName, setter, paramName, opt.GoType(), callName)
		p("c.opt_[%q] = %s\n", opt.name, paramName)
		p("return c\n")
		p("}\n")
	}

	if meth.supportsMedia() {
		p("func (c *%s) Media(r io.Reader) *%s {\n", callName, callName)
		p("c.media_ = r\n")
		p("return c\n")
		p("}\n")
	}

	pn("\nfunc (c *%s) Do() (%serror) {", callName, retTypeComma)

	nilRet := ""
	if retTypeComma != "" {
		nilRet = "nil, "
	}
	pn("var body io.Reader = nil")
	hasContentType := false
	httpMethod := jstr(meth.m, "httpMethod")
	if ba := args.bodyArg(); ba != nil && httpMethod != "GET" {
		style := "WithoutDataWrapper"
		if a.needsDataWrapper() {
			style = "WithDataWrapper"
		}
		pn("body, err := googleapi.%s.JSONReader(c.%s)", style, ba.goname)
		pn("if err != nil { return %serr }", nilRet)
		pn(`ctype := "application/json"`)
		hasContentType = true
	}
	pn("params := make(url.Values)")
	// Set this first. if they override it, though, might be gross.  We don't expect
	// XML replies elsewhere.  TODO(bradfitz): hide this option in the generated code?
	pn(`params.Set("alt", "json")`)
	for _, p := range meth.RequiredQueryParams() {
		pn("params.Set(%q, fmt.Sprintf(\"%%v\", c.%s))", p.name, p.goCallFieldName())
	}
	for _, p := range meth.RequiredRepeatedQueryParams() {
		pn("for _, v := range c.%s { params.Add(%q, fmt.Sprintf(\"%%v\", v)) }",
			p.name, p.name)
	}
	for _, p := range meth.OptParams() {
		pn("if v, ok := c.opt_[%q]; ok { params.Set(%q, fmt.Sprintf(\"%%v\", v)) }",
			p.name, p.name)
	}

	p("urls := googleapi.ResolveRelative(c.s.BasePath, %q)\n", jstr(meth.m, "path"))
	if meth.supportsMedia() {
		pn("if c.media_ != nil {")
		// Hack guess, since we get a 404 otherwise:
		//pn("urls = googleapi.ResolveRelative(%q, %q)", a.apiBaseURL(), meth.mediaPath())
		// Further hack.  Discovery doc is wrong?
		pn("urls = strings.Replace(urls, %q, %q, 1)", "https://www.googleapis.com/", "https://www.googleapis.com/upload/")
		pn(`params.Set("uploadType", "multipart")`)
		pn("}")
	}
	pn("urls += \"?\" + params.Encode()")
	if meth.supportsMedia() && httpMethod != "GET" {
		if !hasContentType { // Support mediaUpload but no ctype set.
			pn("body = new(bytes.Buffer)")
			pn(`ctype := "application/json"`)
			hasContentType = true
		}
		pn("contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)")
	}
	pn("req, _ := http.NewRequest(%q, urls, body)", httpMethod)
	// Replace param values after NewRequest to avoid reencoding them.
	// E.g. Cloud Storage API requires '%2F' in entity param to be kept, but url.Parse replaces it with '/'.
	for _, arg := range args.forLocation("path") {
		pn(`req.URL.Path = strings.Replace(req.URL.Path, "{%s}", %s, 1)`, arg.apiname, arg.cleanExpr("c."))
	}
	// Set opaque to avoid encoding of the parameters in the URL path.
	pn("googleapi.SetOpaque(req.URL)")

	if meth.supportsMedia() {
		pn("if hasMedia_ { req.ContentLength = contentLength_ }")
	}
	if hasContentType {
		pn(`req.Header.Set("Content-Type", ctype)`)
	}
	pn(`req.Header.Set("User-Agent", "google-api-go-client/` + goGenVersion + `")`)
	pn("res, err := c.s.client.Do(req);")
	pn("if err != nil { return %serr }", nilRet)
	pn("defer googleapi.CloseBody(res)")
	pn("if err := googleapi.CheckResponse(res); err != nil { return %serr }", nilRet)
	if retTypeComma == "" {
		pn("return nil")
	} else {
		pn("ret := new(%s)", responseType(a, meth.m)[1:])
		pn("if err := json.NewDecoder(res.Body).Decode(ret); err != nil { return nil, err }")
		pn("return ret, nil")
	}

	bs, _ := json.MarshalIndent(meth.m, "\t// ", "  ")
	pn("// %s\n", string(bs))
	pn("}")
}

type Param struct {
	method        *Method
	name          string
	m             map[string]interface{}
	callFieldName string // empty means to use the default
}

func (p *Param) IsRequired() bool {
	v, _ := p.m["required"].(bool)
	return v
}

func (p *Param) IsRepeated() bool {
	v, _ := p.m["repeated"].(bool)
	return v
}

func (p *Param) Location() string {
	return p.m["location"].(string)
}

func (p *Param) GoType() string {
	typ, format := jstr(p.m, "type"), jstr(p.m, "format")
	if typ == "string" && strings.Contains(format, "int") && p.Location() != "query" {
		panic("unexpected int parameter encoded as string, not in query: " + p.name)
	}
	t, ok := simpleTypeConvert(typ, format)
	if !ok {
		panic("failed to convert parameter type " + fmt.Sprintf("type=%q, format=%q", typ, format))
	}
	return t
}

// goCallFieldName returns the name of this parameter's field in a
// method's "Call" struct.
func (p *Param) goCallFieldName() string {
	if p.callFieldName != "" {
		return p.callFieldName
	}
	return validGoIdentifer(p.name)
}

// APIMethods returns top-level ("API-level") methods. They don't have an associated resource.
func (a *API) APIMethods() []*Method {
	meths := []*Method{}
	methMap := jobj(a.m, "methods")
	for _, name := range sortedKeys(methMap) {
		mi := methMap[name]
		meths = append(meths, &Method{
			api:  a,
			r:    nil, // to be explicit
			name: name,
			m:    mi.(map[string]interface{}),
		})
	}
	return meths
}

func (a *API) Resources(m map[string]interface{}, p string) []*Resource {
	res := []*Resource{}
	resMap := jobj(m, "resources")
	for _, rname := range sortedKeys(resMap) {
		rmi := resMap[rname]
		rm := rmi.(map[string]interface{})
		res = append(res, &Resource{a, rname, p, rm, a.Resources(rm, fmt.Sprintf("%s.%s", p, rname))})
	}
	return res
}

func resolveRelative(basestr, relstr string) string {
	u, err := url.Parse(basestr)
	if err != nil {
		panicf("Error parsing base URL %q: %v", basestr, err)
	}
	rel, err := url.Parse(relstr)
	if err != nil {
		panicf("Error parsing relative URL %q: %v", relstr, err)
	}
	u = u.ResolveReference(rel)
	return u.String()
}

func (meth *Method) NewArguments() (args *arguments) {
	args = &arguments{
		method: meth,
		m:      make(map[string]*argument),
	}
	po, ok := meth.m["parameterOrder"].([]interface{})
	if ok {
		for _, poi := range po {
			pname := poi.(string)
			arg := meth.NewArg(pname, meth.NamedParam(pname))
			args.AddArg(arg)
		}
	}
	if ro := jobj(meth.m, "request"); ro != nil {
		args.AddArg(meth.NewBodyArg(ro))
	}
	return
}

func (meth *Method) NewBodyArg(m map[string]interface{}) *argument {
	reftype := jstr(m, "$ref")
	return &argument{
		goname:   validGoIdentifer(strings.ToLower(reftype)),
		apiname:  "REQUEST",
		gotype:   "*" + reftype,
		apitype:  reftype,
		location: "body",
	}
}

func (meth *Method) NewArg(apiname string, p *Param) *argument {
	m := p.m
	apitype := jstr(m, "type")
	des := jstr(m, "description")
	goname := validGoIdentifer(apiname) // but might be changed later, if conflicts
	if strings.Contains(des, "identifier") && !strings.HasSuffix(strings.ToLower(goname), "id") {
		goname += "id" // yay
		p.callFieldName = goname
	}
	gotype := mustSimpleTypeConvert(apitype, jstr(m, "format"))
	if p.IsRepeated() {
		gotype = "[]" + gotype
	}
	return &argument{
		apiname:  apiname,
		apitype:  apitype,
		goname:   goname,
		gotype:   gotype,
		location: jstr(m, "location"),
	}
}

type argument struct {
	method           *Method
	apiname, apitype string
	goname, gotype   string
	location         string // "path", "query", "body"
}

func (a *argument) String() string {
	return a.goname + " " + a.gotype
}

func (a *argument) cleanExpr(prefix string) string {
	switch a.gotype {
	case "[]string":
		log.Printf("TODO(bradfitz): only including the first parameter in path query.")
		return "url.QueryEscape(" + prefix + a.goname + "[0])"
	case "string":
		return "url.QueryEscape(" + prefix + a.goname + ")"
	case "integer", "int64":
		return "strconv.FormatInt(" + prefix + a.goname + ", 10)"
	case "uint64":
		return "strconv.FormatUint(" + prefix + a.goname + ", 10)"
	}
	log.Panicf("unknown type: apitype=%q, gotype=%q", a.apitype, a.gotype)
	return ""
}

// arguments are the arguments that a method takes
type arguments struct {
	l      []*argument
	m      map[string]*argument
	method *Method
}

func (args *arguments) forLocation(loc string) []*argument {
	matches := make([]*argument, 0)
	for _, arg := range args.l {
		if arg.location == loc {
			matches = append(matches, arg)
		}
	}
	return matches
}

func (args *arguments) bodyArg() *argument {
	for _, arg := range args.l {
		if arg.location == "body" {
			return arg
		}
	}
	return nil
}

func (args *arguments) AddArg(arg *argument) {
	n := 1
	oname := arg.goname
	for {
		_, present := args.m[arg.goname]
		if !present {
			args.m[arg.goname] = arg
			args.l = append(args.l, arg)
			return
		}
		n++
		arg.goname = fmt.Sprintf("%s%d", oname, n)
	}
}

func (a *arguments) String() string {
	var buf bytes.Buffer
	for i, arg := range a.l {
		if i != 0 {
			buf.Write([]byte(", "))
		}
		buf.Write([]byte(arg.String()))
	}
	return buf.String()
}

func asComment(pfx, c string) string {
	var buf bytes.Buffer
	const maxLen = 70
	removeNewlines := func(s string) string {
		return strings.Replace(s, "\n", "\n"+pfx+"// ", -1)
	}
	for len(c) > 0 {
		line := c
		if len(line) < maxLen {
			fmt.Fprintf(&buf, "%s// %s\n", pfx, removeNewlines(line))
			break
		}
		line = line[:maxLen]
		si := strings.LastIndex(line, " ")
		if si != -1 {
			line = line[:si]
		}
		fmt.Fprintf(&buf, "%s// %s\n", pfx, removeNewlines(line))
		c = c[len(line):]
		if si != -1 {
			c = c[1:]
		}
	}
	return buf.String()
}

func simpleTypeConvert(apiType, format string) (gotype string, ok bool) {
	// From http://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.1
	switch apiType {
	case "boolean":
		gotype = "bool"
	case "string":
		gotype = "string"
		switch format {
		case "int64", "uint64", "int32", "uint32":
			gotype = format
		}
	case "number":
		gotype = "float64"
	case "integer":
		gotype = "int64"
	case "any":
		gotype = "interface{}"
	}
	return gotype, gotype != ""
}

func mustSimpleTypeConvert(apiType, format string) string {
	if gotype, ok := simpleTypeConvert(apiType, format); ok {
		return gotype
	}
	panic(fmt.Sprintf("failed to simpleTypeConvert(%q, %q)", apiType, format))
}

func (a *API) goTypeOfJsonObject(outerName, memberName string, m map[string]interface{}) (string, error) {
	apitype := jstr(m, "type")
	switch apitype {
	case "array":
		items := jobj(m, "items")
		if items == nil {
			return "", errors.New("no items but type was array")
		}
		if ref := jstr(items, "$ref"); ref != "" {
			return "[]*" + ref, nil // TODO: wrong; delete this whole function
		}
		if atype := jstr(items, "type"); atype != "" {
			return "[]" + mustSimpleTypeConvert(atype, jstr(items, "format")), nil
		}
		return "", errors.New("unsupported 'array' type")
	case "object":
		return "*" + outerName + "_" + memberName, nil
		//return "", os.NewError("unsupported 'object' type")
	}
	return mustSimpleTypeConvert(apitype, jstr(m, "format")), nil
}

func responseType(api *API, m map[string]interface{}) string {
	ro := jobj(m, "response")
	if ro != nil {
		if ref := jstr(ro, "$ref"); ref != "" {
			if s := api.schemas[ref]; s != nil {
				return "*" + s.GoName()
			}
			return "*" + ref
		}
	}
	return ""
}

// initialCap returns the identifier with a leading capital letter.
// it also maps "foo-bar" to "FooBar".
func initialCap(ident string) string {
	if ident == "" {
		panic("blank identifier")
	}
	return depunct(ident, true)
}

func validGoIdentifer(ident string) string {
	id := depunct(ident, false)
	switch id {
	case "break", "default", "func", "interface", "select",
		"case", "defer", "go", "map", "struct",
		"chan", "else", "goto", "package", "switch",
		"const", "fallthrough", "if", "range", "type",
		"continue", "for", "import", "return", "var":
		return id + "_"
	}
	return id
}

// depunct removes '-', '.', '$', '/' from identifers, making the
// following character uppercase
func depunct(ident string, needCap bool) string {
	var buf bytes.Buffer
	for _, c := range ident {
		if c == '-' || c == '.' || c == '$' || c == '/' {
			needCap = true
			continue
		}
		if needCap {
			c = unicode.ToUpper(c)
			needCap = false
		}
		buf.WriteByte(byte(c))
	}
	return buf.String()

}

func prettyJSON(m map[string]interface{}) string {
	bs, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Sprintf("[JSON error %v on %#v]", err, m)
	}
	return string(bs)
}

func jstr(m map[string]interface{}, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func sortedKeys(m map[string]interface{}) (keys []string) {
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return
}

func jobj(m map[string]interface{}, key string) map[string]interface{} {
	if m, ok := m[key].(map[string]interface{}); ok {
		return m
	}
	return nil
}

func jstrlist(m map[string]interface{}, key string) []string {
	si, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	sl := make([]string, 0)
	for _, si := range si {
		sl = append(sl, si.(string))
	}
	return sl
}
