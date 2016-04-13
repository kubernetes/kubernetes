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
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

const googleDiscoveryURL = "https://www.googleapis.com/discovery/v1/apis"

var (
	apiToGenerate = flag.String("api", "*", "The API ID to generate, like 'tasks:v1'. A value of '*' means all.")
	useCache      = flag.Bool("cache", true, "Use cache of discovered Google API discovery documents.")
	genDir        = flag.String("gendir", "", "Directory to use to write out generated Go files")
	build         = flag.Bool("build", false, "Compile generated packages.")
	install       = flag.Bool("install", false, "Install generated packages.")
	apisURL       = flag.String("discoveryurl", googleDiscoveryURL, "URL to root discovery document")

	publicOnly = flag.Bool("publiconly", true, "Only build public, released APIs. Only applicable for Google employees.")

	jsonFile       = flag.String("api_json_file", "", "If non-empty, the path to a local file on disk containing the API to generate. Exclusive with setting --api.")
	output         = flag.String("output", "", "(optional) Path to source output file. If not specified, the API name and version are used to construct an output path (e.g. tasks/v1).")
	apiPackageBase = flag.String("api_pkg_base", "google.golang.org/api", "Go package prefix to use for all generated APIs.")
	baseURL        = flag.String("base_url", "", "(optional) Override the default service API URL. If empty, the service's root URL will be used.")
	headerPath     = flag.String("header_path", "", "If non-empty, prepend the contents of this file to generated services.")

	contextHTTPPkg = flag.String("ctxhttp_pkg", "golang.org/x/net/context/ctxhttp", "Go package path of the 'ctxhttp' package.")
	contextPkg     = flag.String("context_pkg", "golang.org/x/net/context", "Go package path of the 'context' package.")
	gensupportPkg  = flag.String("gensupport_pkg", "google.golang.org/api/gensupport", "Go package path of the 'api/gensupport' support package.")
	googleapiPkg   = flag.String("googleapi_pkg", "google.golang.org/api/googleapi", "Go package path of the 'api/googleapi' support package.")
)

// commonParamsWhitelist is the list of common API parameters that should be exposed to users via method Call objects.
var commonParamsWhitelist = []string{"quotaUser", "userIp"}

func commonParamWhitelisted(name string) bool {
	for _, whitelisted := range commonParamsWhitelist {
		if name == whitelisted {
			return true
		}
	}
	return false
}

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

	forceJSON     []byte // if non-nil, the JSON schema file. else fetched.
	usedNames     namePool
	schemas       map[string]*Schema // apiName -> schema
	responseTypes map[string]bool

	p  func(format string, args ...interface{}) // print raw
	pn func(format string, args ...interface{}) // print with newline
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

func (all *AllAPIs) addAPI(api string) {
	parts := strings.Split(api, ":")
	if len(parts) != 2 {
		panicf("malformed API name: %q", api)
	}
	apiName := parts[0]
	apiVersion := parts[1]
	all.Items = append(all.Items, &API{
		ID:            api,
		Name:          apiName,
		Version:       apiVersion,
		DiscoveryLink: fmt.Sprintf("./apis/%s/%s/rest", apiName, apiVersion),
	})
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
	if *jsonFile != "" {
		// Return true early, before calling a.JSONFile()
		// which will require a GOPATH be set.  This is for
		// integration with Google's build system genrules
		// where there is no GOPATH.
		return true
	}
	// Skip this API if we're in cached mode and the files don't exist on disk.
	if *useCache {
		if _, err := os.Stat(a.JSONFile()); os.IsNotExist(err) {
			return false
		}
	}
	return *apiToGenerate == "*" || *apiToGenerate == a.ID
}

func getAPIs() []*API {
	if *jsonFile != "" {
		return getAPIsFromFile()
	}
	var all AllAPIs
	var disco []byte
	apiListFile := filepath.Join(genDirRoot(), "api-list.json")
	if *useCache {
		if !*publicOnly {
			log.Fatalf("-cached=true not compatible with -publiconly=false")
		}
		var err error
		disco, err = ioutil.ReadFile(apiListFile)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		disco = slurpURL(*apisURL)
		if *publicOnly {
			if err := writeFile(apiListFile, disco); err != nil {
				log.Fatal(err)
			}
		}
	}
	if err := json.Unmarshal(disco, &all); err != nil {
		log.Fatalf("error decoding JSON in %s: %v", *apisURL, err)
	}
	if !*publicOnly && *apiToGenerate != "*" {
		all.addAPI(*apiToGenerate)
	}
	if *apiToGenerate == "*" && *apisURL == googleDiscoveryURL {
		// Force generation of compute:beta which doesn't appear in the
		// directory for some reason, but they've asked us to include it.
		all.addAPI("compute:beta")
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
	if err == nil && (bytes.Equal(existing, contents) || basicallyEqual(existing, contents)) {
		return nil
	}
	outdir := filepath.Dir(file)
	if err = os.MkdirAll(outdir, 0755); err != nil {
		return fmt.Errorf("failed to Mkdir %s: %v", outdir, err)
	}
	return ioutil.WriteFile(file, contents, 0644)
}

var ignoreLines = regexp.MustCompile(`(?m)^\s+"(?:etag|revision)": ".+\n`)

// basicallyEqual reports whether a and b are equal except for boring
// differences like ETag updates.
func basicallyEqual(a, b []byte) bool {
	return ignoreLines.Match(a) && ignoreLines.Match(b) &&
		bytes.Equal(ignoreLines.ReplaceAll(a, nil), ignoreLines.ReplaceAll(b, nil))
}

func slurpURL(urlStr string) []byte {
	if *useCache {
		log.Fatalf("Invalid use of slurpURL in cached mode for URL %s", urlStr)
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

// oddVersionRE matches unusual API names like directory_v1.
var oddVersionRE = regexp.MustCompile(`^(.+)_(v[\d\.]+)$`)

// renameVersion conditionally rewrites the provided version such
// that the final path component of the import path doesn't look
// like a Go identifier. This keeps the consistency that import paths
// for the generated Go packages look like:
//     google.golang.org/api/NAME/v<version>
// and have package NAME.
// See https://github.com/google/google-api-go-client/issues/78
func renameVersion(version string) string {
	if version == "alpha" || version == "beta" {
		return "v0." + version
	}
	if m := oddVersionRE.FindStringSubmatch(version); m != nil {
		return m[1] + "/" + m[2]
	}
	return version
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

func genDirRoot() string {
	if *genDir != "" {
		return *genDir
	}
	paths := filepath.SplitList(os.Getenv("GOPATH"))
	if len(paths) == 0 {
		log.Fatalf("No GOPATH set.")
	}
	return filepath.Join(paths[0], "src", "google.golang.org", "api")
}

func (a *API) SourceDir() string {
	return filepath.Join(genDirRoot(), a.Package(), renameVersion(a.Version))
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
	return fmt.Sprintf("%s/%s/%s", *apiPackageBase, a.Package(), renameVersion(a.Version))
}

// GetName returns a free top-level function/type identifier in the package.
// It tries to return your preferred match if it's free.
func (a *API) GetName(preferred string) string {
	return a.usedNames.Get(preferred)
}

func (a *API) apiBaseURL() string {
	var base, rel string
	switch {
	case *baseURL != "":
		base, rel = *baseURL, jstr(a.m, "basePath")
	case a.RootURL != "":
		base, rel = a.RootURL, a.ServicePath
	default:
		base, rel = *apisURL, jstr(a.m, "basePath")
	}
	return resolveRelative(base, rel)
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
	if *useCache {
		slurp, err := ioutil.ReadFile(a.JSONFile())
		if err != nil {
			log.Fatal(err)
		}
		return slurp
	}
	return slurpURL(a.DiscoveryURL())
}

func (a *API) JSONFile() string {
	return filepath.Join(a.SourceDir(), a.Package()+"-api.json")
}

func (a *API) WriteGeneratedCode() error {
	genfilename := *output
	if genfilename == "" {
		if err := writeFile(a.JSONFile(), a.jsonBytes()); err != nil {
			return err
		}
		outdir := a.SourceDir()
		err := os.MkdirAll(outdir, 0755)
		if err != nil {
			return fmt.Errorf("failed to Mkdir %s: %v", outdir, err)
		}
		pkg := a.Package()
		genfilename = filepath.Join(outdir, pkg+"-gen.go")
	}

	code, err := a.GenerateCode()
	errw := writeFile(genfilename, code)
	if err == nil {
		err = errw
	}
	return err
}

var docsLink string

func (a *API) GenerateCode() ([]byte, error) {
	pkg := a.Package()

	a.m = make(map[string]interface{})
	m := a.m
	jsonBytes := a.jsonBytes()
	err := json.Unmarshal(jsonBytes, &a.m)
	if err != nil {
		return nil, err
	}
	// Because the Discovery JSON may not have all the fields populated that the actual
	// API JSON has (e.g. rootUrl and servicePath), the API should be repopulated from
	// the JSON here.
	if err := json.Unmarshal(jsonBytes, a); err != nil {
		return nil, err
	}

	// Buffer the output in memory, for gofmt'ing later.
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
	wf := func(path string) error {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		_, err = io.Copy(&buf, f)
		return err
	}

	p, pn := a.p, a.pn
	reslist := a.Resources(a.m, "")

	if *headerPath != "" {
		if err := wf(*headerPath); err != nil {
			return nil, err
		}
	}

	pn("// Package %s provides access to the %s.", pkg, jstr(m, "title"))
	docsLink = jstr(m, "documentationLink")
	if docsLink != "" {
		pn("//")
		pn("// See %s", docsLink)
	}
	pn("//\n// Usage example:")
	pn("//")
	pn("//   import %q", a.Target())
	pn("//   ...")
	pn("//   %sService, err := %s.New(oauthHttpClient)", pkg, pkg)

	pn("package %s // import %q", pkg, a.Target())
	p("\n")
	pn("import (")
	for _, imp := range []struct {
		pkg   string
		lname string
	}{
		{"bytes", ""},
		{"encoding/json", ""},
		{"errors", ""},
		{"fmt", ""},
		{"io", ""},
		{"net/http", ""},
		{"net/url", ""},
		{"strconv", ""},
		{"strings", ""},
		{*contextHTTPPkg, "ctxhttp"},
		{*contextPkg, "context"},
		{*gensupportPkg, "gensupport"},
		{*googleapiPkg, "googleapi"},
	} {
		if imp.lname == "" {
			pn("  %q", imp.pkg)
		} else {
			pn("  %s %q", imp.lname, imp.pkg)
		}
	}
	pn(")")
	pn("\n// Always reference these packages, just in case the auto-generated code")
	pn("// below doesn't.")
	pn("var _ = bytes.NewBuffer")
	pn("var _ = strconv.Itoa")
	pn("var _ = fmt.Sprintf")
	pn("var _ = json.NewDecoder")
	pn("var _ = io.Copy")
	pn("var _ = url.Parse")
	pn("var _ = gensupport.MarshalJSON")
	pn("var _ = googleapi.Version")
	pn("var _ = errors.New")
	pn("var _ = strings.Replace")
	pn("var _ = context.Canceled")
	pn("var _ = ctxhttp.Do")
	pn("")
	pn("const apiId = %q", jstr(m, "id"))
	pn("const apiName = %q", jstr(m, "name"))
	pn("const apiVersion = %q", jstr(m, "version"))
	pn("const basePath = %q", a.apiBaseURL())

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
	pn("\ntype Service struct {")
	pn(" client *http.Client")
	pn(" BasePath string // API endpoint base URL")
	pn(" UserAgent string // optional additional User-Agent fragment")

	for _, res := range reslist {
		pn("\n\t%s\t*%s", res.GoField(), res.GoType())
	}
	pn("}")
	pn("\nfunc (s *Service) userAgent() string {")
	pn(` if s.UserAgent == "" { return googleapi.UserAgent }`)
	pn(` return googleapi.UserAgent + " " + s.UserAgent`)
	pn("}\n")

	for _, res := range reslist {
		res.generateType()
	}

	a.PopulateSchemas()

	a.responseTypes = make(map[string]bool)
	for _, meth := range a.APIMethods() {
		meth.cacheResponseTypes(a)
	}
	for _, res := range reslist {
		res.cacheResponseTypes(a)
	}

	for _, name := range a.sortedSchemaNames() {
		a.schemas[name].writeSchemaCode(a)
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

	a.pn("// OAuth2 scopes used by this API.")
	a.pn("const (")
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
		a.pn("\t%s = %q", ident, scopeName)
	}
	a.p(")\n\n")
}

func scopeIdentifierFromURL(urlStr string) string {
	const prefix = "https://www.googleapis.com/auth/"
	if !strings.HasPrefix(urlStr, prefix) {
		const https = "https://"
		if !strings.HasPrefix(urlStr, https) {
			log.Fatalf("Unexpected oauth2 scope %q doesn't start with %q", urlStr, https)
		}
		ident := validGoIdentifer(depunct(urlStr[len(https):], true)) + "Scope"
		return ident
	}
	ident := validGoIdentifer(initialCap(urlStr[len(prefix):])) + "Scope"
	return ident
}

type Schema struct {
	api *API
	m   map[string]interface{} // original JSON map

	typ *Type // lazily populated by Type

	apiName      string // the native API-defined name of this type
	goName       string // lazily populated by GoName
	goReturnType string // lazily populated by GoReturnType
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

func (p *Property) Default() string {
	return jstr(p.m, "default")
}

func (p *Property) Description() string {
	return jstr(p.m, "description")
}

func (p *Property) Enum() ([]string, bool) {
	if enums := jstrlist(p.m, "enum"); enums != nil {
		return enums, true
	}
	// Check if this has an array of string enums.
	if items := jobj(p.m, "items"); items != nil {
		if enums := jstrlist(items, "enum"); enums != nil && jstr(items, "type") == "string" {
			return enums, true
		}
	}
	return nil, false
}

func (p *Property) EnumDescriptions() []string {
	if desc := jstrlist(p.m, "enumDescriptions"); desc != nil {
		return desc
	}
	// Check if this has an array of string enum descriptions.
	if items := jobj(p.m, "items"); items != nil {
		if desc := jstrlist(items, "enumDescriptions"); desc != nil {
			return desc
		}
	}
	return nil
}

func (p *Property) Pattern() (string, bool) {
	if s, ok := p.m["pattern"].(string); ok {
		return s, true
	}
	return "", false
}

// A FieldName uniquely identifies a field within a Schema struct for an API.
type fieldName struct {
	api    string // The ID of an API.
	schema string // The Go name of a Schema struct.
	field  string // The Go name of a field.
}

// pointerFields is a list of fields that should use a pointer type.
// This makes it possible to distinguish between a field being unset vs having
// an empty value.
var pointerFields = []fieldName{
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "BoolValue"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "DoubleValue"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "Int64Value"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "StringValue"},
	{api: "compute:v1", schema: "MetadataItems", field: "Value"},
	{api: "content:v2", schema: "AccountUser", field: "Admin"},
	{api: "datastore:v1beta2", schema: "Property", field: "BlobKeyValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "BlobValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "BooleanValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "DateTimeValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "DoubleValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "Indexed"},
	{api: "datastore:v1beta2", schema: "Property", field: "IntegerValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "StringValue"},
	{api: "genomics:v1beta2", schema: "Dataset", field: "IsPublic"},
	{api: "tasks:v1", schema: "Task", field: "Completed"},
	{api: "youtube:v3", schema: "ChannelSectionSnippet", field: "Position"},
}

// forcePointerType reports whether p should be represented as a pointer type in its parent schema struct.
func (p *Property) forcePointerType() bool {
	if p.UnfortunateDefault() {
		return true
	}

	name := fieldName{api: p.s.api.ID, schema: p.s.GoName(), field: p.GoName()}
	for _, pf := range pointerFields {
		if pf == name {
			return true
		}
	}
	return false
}

// UnfortunateDefault reports whether p may be set to a zero value, but has a non-zero default.
func (p *Property) UnfortunateDefault() bool {
	switch p.Type().AsGo() {
	default:
		return false

	case "bool":
		return p.Default() == "true"

	case "string":
		if p.Default() == "" {
			return false
		}
		// String fields are considered to "allow" a zero value if either:
		//  (a) they are an enum, and one of the permitted enum values is the empty string, or
		//  (b) they have a validation pattern which matches the empty string.
		pattern, hasPat := p.Pattern()
		enum, hasEnum := p.Enum()
		if hasPat && hasEnum {
			log.Printf("Encountered enum property which also has a pattern: %#v", p)
			return false // don't know how to handle this, so ignore.
		}
		return (hasPat && emptyPattern(pattern)) ||
			(hasEnum && emptyEnum(enum))

	case "float64", "int64", "uint64", "int32", "uint32":
		if p.Default() == "" {
			return false
		}
		if f, err := strconv.ParseFloat(p.Default(), 64); err == nil {
			return f != 0.0
		}
		// The default value has an unexpected form.  Whatever it is, it's non-zero.
		return true
	}
}

// emptyPattern reports whether a pattern matches the empty string.
func emptyPattern(pattern string) bool {
	if re, err := regexp.Compile(pattern); err == nil {
		return re.MatchString("")
	}
	log.Printf("Encountered bad pattern: %s", pattern)
	return false
}

// emptyEnum reports whether a property enum list contains the empty string.
func emptyEnum(enum []string) bool {
	for _, val := range enum {
		if val == "" {
			return true
		}
	}
	return false
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
	if typ, ok := t.MapType(); ok {
		return typ
	}
	isAny := t.IsAny()
	if t.IsStruct() || isAny {
		if apiName, ok := t.m["_apiName"].(string); ok {
			s := t.api.schemas[apiName]
			if s == nil {
				panic(fmt.Sprintf("in Type.AsGo, _apiName of %q didn't point to a valid schema; json: %s",
					apiName, prettyJSON(t.m)))
			}
			if isAny {
				return s.GoName() // interface type; no pointer.
			}
			if v := jobj(s.m, "variant"); v != nil {
				return s.GoName()
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

func (t *Type) IsAny() bool {
	if t.apiType() == "object" {
		props := jobj(t.m, "additionalProperties")
		if props != nil && jstr(props, "type") == "any" {
			return true
		}
	}
	return false
}

func (t *Type) Reference() (apiName string, ok bool) {
	apiName = jstr(t.m, "$ref")
	ok = apiName != ""
	return
}

func (t *Type) IsMap() bool {
	_, ok := t.MapType()
	return ok
}

// MapType checks if the current node is a map and if true, it returns the Go type for the map, such as map[string]string.
func (t *Type) MapType() (typ string, ok bool) {
	props := jobj(t.m, "additionalProperties")
	if props == nil {
		return "", false
	}
	s := jstr(props, "type")
	if s == "any" {
		return "", false
	}
	if s == "string" {
		return "map[string]string", true
	}
	if s != "array" {
		if s == "" { // Check for reference
			s = jstr(props, "$ref")
			if s != "" {
				return "map[string]" + s, true
			}
		}
		if s == "any" {
			return "map[string]interface{}", true
		}
		log.Printf("Warning: found map to type %q which is not implemented yet.", s)
		return "", false
	}
	items := jobj(props, "items")
	if items == nil {
		return "", false
	}
	s = jstr(items, "type")
	if s != "string" {
		if s == "" { // Check for reference
			s = jstr(items, "$ref")
			if s != "" {
				return "map[string][]" + s, true
			}
		}
		log.Printf("Warning: found map of arrays of type %q which is not implemented yet.", s)
		return "", false
	}
	return "map[string][]string", true
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
		if name, ok := s.Type().MapType(); ok {
			s.goName = name
		} else {
			s.goName = s.api.GetName(initialCap(s.apiName))
		}
	}
	return s.goName
}

// GoReturnType returns the Go type to use as the return type.
// If a type is a struct, it will return *StructType,
// for a map it will return map[string]ValueType,
// for (not yet supported) slices it will return []ValueType.
func (s *Schema) GoReturnType() string {
	if s.goReturnType == "" {
		if s.Type().IsMap() {
			s.goReturnType = s.GoName()
		} else {
			s.goReturnType = "*" + s.GoName()
		}
	}
	return s.goReturnType
}

func (s *Schema) writeSchemaCode(api *API) {
	if s.Type().IsAny() {
		s.api.pn("\ntype %s interface{}", s.GoName())
		return
	}
	if s.Type().IsStruct() && !s.Type().IsMap() {
		s.writeSchemaStruct(api)
		return
	}

	if _, ok := s.Type().ArrayType(); ok {
		log.Printf("TODO writeSchemaCode for arrays for %s", s.GoName())
		return
	}

	if destSchema, ok := s.Type().ReferenceSchema(); ok {
		// Convert it to a struct using embedding.
		s.api.pn("\ntype %s struct {", s.GoName())
		s.api.pn(" %s", destSchema.GoName())
		s.api.pn("}")
		return
	}

	if s.Type().IsSimple() {
		apitype := jstr(s.m, "type")
		typ := mustSimpleTypeConvert(apitype, jstr(s.m, "format"))
		s.api.pn("\ntype %s %s", s.GoName(), typ)
		return
	}

	if s.Type().IsMap() {
		return
	}

	fmt.Fprintf(os.Stderr, "in writeSchemaCode, schema is: %s", prettyJSON(s.m))
	panicf("writeSchemaCode: unsupported type for schema %q", s.apiName)
}

func (s *Schema) writeVariant(api *API, v map[string]interface{}) {
	s.api.p("\ntype %s map[string]interface{}\n\n", s.GoName())

	// Write out the "Type" method that identifies the variant type.
	s.api.pn("func (t %s) Type() string {", s.GoName())
	s.api.pn("  return googleapi.VariantType(t)")
	s.api.p("}\n\n")

	// Write out helper methods to convert each possible variant.
	for _, m := range jobjlist(v, "map") {
		val := jstr(m, "type_value")
		reftype := jstr(m, "$ref")
		if val == "" && reftype == "" {
			log.Printf("TODO variant %s ref %s not yet supported.", val, reftype)
			continue
		}

		_, ok := api.schemas[reftype]
		if !ok {
			log.Printf("TODO variant %s ref %s not yet supported.", val, reftype)
			continue
		}

		s.api.pn("func (t %s) %s() (r %s, ok bool) {", s.GoName(), initialCap(val), reftype)
		s.api.pn(" if t.Type() != %q {", initialCap(val))
		s.api.pn("  return r, false")
		s.api.pn(" }")
		s.api.pn(" ok = googleapi.ConvertVariant(map[string]interface{}(t), &r)")
		s.api.pn(" return r, ok")
		s.api.p("}\n\n")
	}
}

func (s *Schema) Description() string {
	return jstr(s.m, "description")
}

func (s *Schema) writeSchemaStruct(api *API) {
	if v := jobj(s.m, "variant"); v != nil {
		s.writeVariant(api, v)
		return
	}
	s.api.p("\n")
	des := s.Description()
	if des != "" {
		s.api.p("%s", asComment("", fmt.Sprintf("%s: %s", s.GoName(), des)))
	}
	s.api.pn("type %s struct {", s.GoName())

	np := new(namePool)
	forceSendName := np.Get("ForceSendFields")
	if s.isResponseType() {
		np.Get("ServerResponse") // reserve the name
	}

	firstFieldName := "" // used to store a struct field name for use in documentation.
	for i, p := range s.properties() {
		if i > 0 {
			s.api.p("\n")
		}
		pname := np.Get(p.GoName())
		des := p.Description()
		if des != "" {
			s.api.p("%s", asComment("\t", fmt.Sprintf("%s: %s", pname, des)))
		}
		addFieldValueComments(s.api.p, p, "\t", des != "")

		var extraOpt string
		if p.Type().isIntAsString() {
			extraOpt += ",string"
		}

		typ := p.Type().AsGo()
		if p.forcePointerType() {
			typ = "*" + typ
		}

		s.api.pn(" %s %s `json:\"%s,omitempty%s\"`", pname, typ, p.APIName(), extraOpt)
		if firstFieldName == "" {
			firstFieldName = pname
		}
	}

	if s.isResponseType() {
		if firstFieldName != "" {
			s.api.p("\n")
		}
		s.api.p("%s", asComment("\t", "ServerResponse contains the HTTP response code and headers from the server."))
		s.api.pn(" googleapi.ServerResponse `json:\"-\"`")
	}

	if firstFieldName == "" {
		// There were no fields in the struct, so there is no point
		// adding any custom JSON marshaling code.
		s.api.pn("}")
		return
	}

	commentFmtStr := "%s is a list of field names (e.g. %q) to " +
		"unconditionally include in API requests. By default, fields " +
		"with empty values are omitted from API requests. However, " +
		"any non-pointer, non-interface field appearing in %s will " +
		"be sent to the server regardless of whether the field is " +
		"empty or not. This may be used to include empty fields in " +
		"Patch requests."
	comment := fmt.Sprintf(commentFmtStr, forceSendName, firstFieldName, forceSendName)
	s.api.p("\n")
	s.api.p("%s", asComment("\t", comment))

	s.api.pn("\t%s []string `json:\"-\"`", forceSendName)
	s.api.pn("}")
	s.writeSchemaMarshal(forceSendName)
	return
}

// writeSchemaMarshal writes a custom MarshalJSON function for s, which allows
// fields to be explicitly transmitted by listing them in the field identified
// by forceSendFieldName.
func (s *Schema) writeSchemaMarshal(forceSendFieldName string) {
	s.api.pn("func (s *%s) MarshalJSON() ([]byte, error) {", s.GoName())
	s.api.pn("\ttype noMethod %s", s.GoName())
	// pass schema as methodless type to prevent subsequent calls to MarshalJSON from recursing indefinitely.
	s.api.pn("\traw := noMethod(*s)")
	s.api.pn("\treturn gensupport.MarshalJSON(raw, s.%s)", forceSendFieldName)
	s.api.pn("}")
}

// isResponseType returns true for all types that are used as a response.
func (s *Schema) isResponseType() bool {
	return s.api.responseTypes["*"+s.goName]
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
	pn := r.api.pn
	t := r.GoType()
	pn(fmt.Sprintf("func New%s(s *Service) *%s {", t, t))
	pn("rs := &%s{s : s}", t)
	for _, res := range r.resources {
		pn("rs.%s = New%s(s)", res.GoField(), res.GoType())
	}
	pn("return rs")
	pn("}")

	pn("\ntype %s struct {", t)
	pn(" s *Service")
	for _, res := range r.resources {
		pn("\n\t%s\t*%s", res.GoField(), res.GoType())
	}
	pn("}")

	for _, res := range r.resources {
		res.generateType()
	}
}

func (r *Resource) cacheResponseTypes(api *API) {
	for _, meth := range r.Methods() {
		meth.cacheResponseTypes(api)
	}
	for _, res := range r.resources {
		res.cacheResponseTypes(api)
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

func (m *Method) supportsMediaUpload() bool {
	return jobj(m.m, "mediaUpload") != nil
}

func (m *Method) mediaUploadPath() string {
	return jstr(jobj(jobj(jobj(m.m, "mediaUpload"), "protocols"), "simple"), "path")
}

func (m *Method) supportsMediaDownload() bool {
	if m.supportsMediaUpload() {
		// storage.objects.insert claims support for download in
		// addition to upload but attempting to do so fails.
		// This situation doesn't apply to any other methods.
		return false
	}
	if v, ok := m.m["supportsMediaDownload"].(bool); ok {
		return v
	}
	return false
}

func (m *Method) Params() []*Param {
	if m.params == nil {
		parameters := jobj(m.m, "parameters")
		// We construct the list of Method parameters as the union of
		// * the method parameters from the JSON config, and
		// * a set of parameters that are common across all the methods in the API.
		// We first make a copy of parameters, so that the original method-specific
		// parameter list is left untouched.
		paramMap := make(map[string]interface{})
		for k, v := range parameters {
			paramMap[k] = v
		}

		commonParams := jobj(m.api.m, "parameters")
		for ck, cv := range commonParams {
			if !commonParamWhitelisted(ck) {
				continue
			}
			if _, ok := paramMap[ck]; ok {
				msgTemplate := "parameter %q of method %q in api %q conflicts with common API parameter of the same name."
				log.Printf(msgTemplate, ck, m.name, m.api.ID)
			} else {
				paramMap[ck] = cv
			}
		}

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

func (meth *Method) cacheResponseTypes(api *API) {
	if retType := responseType(api, meth.m); retType != "" && strings.HasPrefix(retType, "*") {
		api.responseTypes[retType] = true
	}
}

// convertMultiParams builds a []string temp variable from a slice
// of non-strings and returns the name of the temp variable.
func convertMultiParams(a *API, param string) string {
	a.pn(" var %v_ []string", param)
	a.pn(" for _, v := range %v {", param)
	a.pn("  %v_ = append(%v_, fmt.Sprint(v))", param, param)
	a.pn(" }")
	return param + "_"
}

// overrideParameterName transforms a hard-coded list of non-idiomatic-Go names into idiomatic Go names.
func overrideParameterName(name string) string {
	// TODO(mcgreevy): make this more general.
	if name == "userIp" {
		return "userIP"
	}
	return name
}

func (meth *Method) generateCode() {
	res := meth.r // may be nil if a top-level method
	a := meth.api
	p, pn := a.p, a.pn

	pn("\n// method id %q:", meth.Id())

	retType := responseType(a, meth.m)
	retTypeComma := retType
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

	pn("\ntype %s struct {", callName)
	pn(" s *Service")
	for _, arg := range args.l {
		if arg.location != "query" {
			pn(" %s %s", arg.goname, arg.gotype)
		}
	}
	pn(" urlParams_ gensupport.URLParams")
	httpMethod := jstr(meth.m, "httpMethod")
	if httpMethod == "GET" {
		pn(" ifNoneMatch_ string")
	}
	if meth.supportsMediaUpload() {
		pn(" media_     io.Reader")
		pn(" resumable_ googleapi.SizeReaderAt")
		pn(" mediaType_ string")
		pn(" protocol_  string")
		pn(" progressUpdater_  googleapi.ProgressUpdater")
	}
	pn(" ctx_ context.Context")
	pn("}")

	p("\n%s", asComment("", methodName+": "+jstr(meth.m, "description")))
	if res != nil {
		if url := canonicalDocsURL[fmt.Sprintf("%v%v/%v", docsLink, res.name, meth.name)]; url != "" {
			pn("// For details, see %v", url)
		}
	}

	var servicePtr string
	if res == nil {
		pn("func (s *Service) %s(%s) *%s {", methodName, args, callName)
		servicePtr = "s"
	} else {
		pn("func (r *%s) %s(%s) *%s {", res.GoType(), methodName, args, callName)
		servicePtr = "r.s"
	}

	pn(" c := &%s{s: %s, urlParams_: make(gensupport.URLParams)}", callName, servicePtr)
	for _, arg := range args.l {
		// TODO(gmlewis): clean up and consolidate this section.
		// See: https://code-review.googlesource.com/#/c/3520/18/google-api-go-generator/gen.go
		if arg.location == "query" {
			switch arg.gotype {
			case "[]string":
				pn(" c.urlParams_.SetMulti(%q, append([]string{}, %v...))", arg.apiname, arg.goname)
			case "string":
				pn(" c.urlParams_.Set(%q, %v)", arg.apiname, arg.goname)
			default:
				if strings.HasPrefix(arg.gotype, "[]") {
					tmpVar := convertMultiParams(a, arg.goname)
					pn(" c.urlParams_.SetMulti(%q, %v)", arg.apiname, tmpVar)
				} else {
					pn(" c.urlParams_.Set(%q, fmt.Sprint(%v))", arg.apiname, arg.goname)
				}
			}
			continue
		}
		if arg.gotype == "[]string" {
			pn(" c.%s = append([]string{}, %s...)", arg.goname, arg.goname) // Make a copy of the []string.
			continue
		}
		pn(" c.%s = %s", arg.goname, arg.goname)
	}
	pn(" return c")
	pn("}")

	for _, opt := range meth.OptParams() {
		if opt.Location() != "query" {
			panicf("optional parameter has unsupported location %q", opt.Location())
		}
		setter := initialCap(overrideParameterName(opt.name))
		des := jstr(opt.m, "description")
		des = strings.Replace(des, "Optional.", "", 1)
		des = strings.TrimSpace(des)
		p("\n%s", asComment("", fmt.Sprintf("%s sets the optional parameter %q: %s", setter, opt.name, des)))
		addFieldValueComments(p, opt, "", true)
		np := new(namePool)
		np.Get("c") // take the receiver's name
		paramName := np.Get(validGoIdentifer(overrideParameterName(opt.name)))
		typePrefix := ""
		if opt.IsRepeated() {
			typePrefix = "..."
		}
		pn("func (c *%s) %s(%s %s%s) *%s {", callName, setter, paramName, typePrefix, opt.GoType(), callName)
		if opt.IsRepeated() {
			if opt.GoType() == "string" {
				pn("c.urlParams_.SetMulti(%q, append([]string{}, %v...))", opt.name, paramName)
			} else {
				tmpVar := convertMultiParams(a, paramName)
				pn(" c.urlParams_.SetMulti(%q, %v)", opt.name, tmpVar)
			}
		} else {
			if opt.GoType() == "string" {
				pn("c.urlParams_.Set(%q, %v)", opt.name, paramName)
			} else {
				pn("c.urlParams_.Set(%q, fmt.Sprint(%v))", opt.name, paramName)
			}
		}
		pn("return c")
		pn("}")
	}

	if meth.supportsMediaUpload() {
		comment := "Media specifies the media to upload in a single chunk. " +
			"At most one of Media and ResumableMedia may be set."
		// TODO(mcgreevy): Ensure that r is always closed before Do returns, and document this.
		// See comments on https://code-review.googlesource.com/#/c/3970/
		p("\n%s", asComment("", comment))
		pn("func (c *%s) Media(r io.Reader) *%s {", callName, callName)
		pn("c.media_ = r")
		pn(`c.protocol_ = "multipart"`)
		pn("return c")
		pn("}")
		comment = "ResumableMedia specifies the media to upload in chunks and can be canceled with ctx. " +
			"At most one of Media and ResumableMedia may be set. " +
			`mediaType identifies the MIME media type of the upload, such as "image/png". ` +
			`If mediaType is "", it will be auto-detected. ` +
			`The provided ctx will supersede any context previously provided to ` +
			`the Context method.`
		p("\n%s", asComment("", comment))
		pn("func (c *%s) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *%s {", callName, callName)
		pn("c.ctx_ = ctx")
		pn("c.resumable_ = io.NewSectionReader(r, 0, size)")
		pn("c.mediaType_ = mediaType")
		pn(`c.protocol_ = "resumable"`)
		pn("return c")
		pn("}")
		comment = "ProgressUpdater provides a callback function that will be called after every chunk. " +
			"It should be a low-latency function in order to not slow down the upload operation. " +
			"This should only be called when using ResumableMedia (as opposed to Media)."
		p("\n%s", asComment("", comment))
		pn("func (c *%s) ProgressUpdater(pu googleapi.ProgressUpdater) *%s {", callName, callName)
		pn(`c.progressUpdater_ = pu`)
		pn("return c")
		pn("}")
	}

	comment := "Fields allows partial responses to be retrieved. " +
		"See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse " +
		"for more information."
	p("\n%s", asComment("", comment))
	pn("func (c *%s) Fields(s ...googleapi.Field) *%s {", callName, callName)
	pn(`c.urlParams_.Set("fields", googleapi.CombineFields(s))`)
	pn("return c")
	pn("}")
	if httpMethod == "GET" {
		// Note that non-GET responses are excluded from supporting If-None-Match.
		// See https://github.com/google/google-api-go-client/issues/107 for more info.
		comment := "IfNoneMatch sets the optional parameter which makes the operation fail if " +
			"the object's ETag matches the given value. This is useful for getting updates " +
			"only after the object has changed since the last request. " +
			"Use googleapi.IsNotModified to check whether the response error from Do " +
			"is the result of In-None-Match."
		p("\n%s", asComment("", comment))
		pn("func (c *%s) IfNoneMatch(entityTag string) *%s {", callName, callName)
		pn(" c.ifNoneMatch_ = entityTag")
		pn(" return c")
		pn("}")
	}

	doMethod := "Do method"
	if meth.supportsMediaDownload() {
		doMethod = "Do and Download methods"
	}
	commentFmtStr := "Context sets the context to be used in this call's %s. " +
		"Any pending HTTP request will be aborted if the provided context is canceled."
	comment = fmt.Sprintf(commentFmtStr, doMethod)
	p("\n%s", asComment("", comment))
	if meth.supportsMediaUpload() {
		comment = "This context will supersede any context previously provided to " +
			"the ResumableMedia method."
		p("%s", asComment("", comment))
	}
	pn("func (c *%s) Context(ctx context.Context) *%s {", callName, callName)
	pn(`c.ctx_ = ctx`)
	pn("return c")
	pn("}")

	pn("\nfunc (c *%s) doRequest(alt string) (*http.Response, error) {", callName)
	pn("var body io.Reader = nil")
	hasContentType := false // Whether ctype has been set.  It is always set in conjunction with body.
	if ba := args.bodyArg(); ba != nil && httpMethod != "GET" {
		style := "WithoutDataWrapper"
		if a.needsDataWrapper() {
			style = "WithDataWrapper"
		}
		pn("body, err := googleapi.%s.JSONReader(c.%s)", style, ba.goname)
		pn("if err != nil { return nil, err }")
		pn(`ctype := "application/json"`)
		hasContentType = true
	}
	pn(`c.urlParams_.Set("alt", alt)`)

	pn("urls := googleapi.ResolveRelative(c.s.BasePath, %q)", jstr(meth.m, "path"))
	if meth.supportsMediaUpload() {
		pn("if c.media_ != nil || c.resumable_ != nil {")
		// Hack guess, since we get a 404 otherwise:
		//pn("urls = googleapi.ResolveRelative(%q, %q)", a.apiBaseURL(), meth.mediaUploadPath())
		// Further hack.  Discovery doc is wrong?
		pn("urls = strings.Replace(urls, %q, %q, 1)", "https://www.googleapis.com/", "https://www.googleapis.com/upload/")
		pn(`c.urlParams_.Set("uploadType", c.protocol_)`)
		pn("}")
	}
	pn("urls += \"?\" + c.urlParams_.Encode()")
	if meth.supportsMediaUpload() && httpMethod != "GET" {
		if !hasContentType {
			pn("body = new(bytes.Buffer)")
			pn(`ctype := "application/json"`)
			hasContentType = true
		}
		pn(`if c.protocol_ != "resumable" && c.media_ != nil {`)
		pn("  cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)")
		pn("  defer cancel()")
		pn("}")
	}
	pn("req, _ := http.NewRequest(%q, urls, body)", httpMethod)
	// Replace param values after NewRequest to avoid reencoding them.
	// E.g. Cloud Storage API requires '%2F' in entity param to be kept, but url.Parse replaces it with '/'.
	argsForLocation := args.forLocation("path")
	if len(argsForLocation) > 0 {
		pn(`googleapi.Expand(req.URL, map[string]string{`)
		for _, arg := range argsForLocation {
			pn(`"%s": %s,`, arg.apiname, arg.exprAsString("c."))
		}
		pn(`})`)
	} else {
		// Just call SetOpaque since we aren't calling Expand
		pn(`googleapi.SetOpaque(req.URL)`)
	}

	if meth.supportsMediaUpload() {
		pn(`if c.protocol_ == "resumable" {`)
		pn(` if c.mediaType_ == "" {`)
		pn("  c.mediaType_ = gensupport.DetectMediaType(c.resumable_)")
		pn(" }")
		pn(` req.Header.Set("X-Upload-Content-Type", c.mediaType_)`)
		pn("}")
	}

	if hasContentType {
		pn(`req.Header.Set("Content-Type", ctype)`)
	}

	pn(`req.Header.Set("User-Agent", c.s.userAgent())`)
	if httpMethod == "GET" {
		pn(`if c.ifNoneMatch_ != "" {`)
		pn(` req.Header.Set("If-None-Match", c.ifNoneMatch_)`)
		pn("}")
	}
	pn("if c.ctx_ != nil {")
	pn(" return ctxhttp.Do(c.ctx_, c.s.client, req)")
	pn("}")
	pn("return c.s.client.Do(req)")
	pn("}")

	if meth.supportsMediaDownload() {
		pn("\n// Download fetches the API endpoint's \"media\" value, instead of the normal")
		pn("// API response value. If the returned error is nil, the Response is guaranteed to")
		pn("// have a 2xx status code. Callers must close the Response.Body as usual.")
		pn("func (c *%s) Download() (*http.Response, error) {", callName)
		pn(`res, err := c.doRequest("media")`)
		pn("if err != nil { return nil, err }")
		pn("if err := googleapi.CheckMediaResponse(res); err != nil {")
		pn("res.Body.Close()")
		pn("return nil, err")
		pn("}")
		pn("return res, nil")
		pn("}")
	}

	mapRetType := strings.HasPrefix(retTypeComma, "map[")
	pn("\n// Do executes the %q call.", jstr(meth.m, "id"))
	if retTypeComma != "" && !mapRetType {
		commentFmtStr := "Exactly one of %v or error will be non-nil. " +
			"Any non-2xx status code is an error. " +
			"Response headers are in either %v.ServerResponse.Header " +
			"or (if a response was returned at all) in error.(*googleapi.Error).Header. " +
			"Use googleapi.IsNotModified to check whether the returned error was because " +
			"http.StatusNotModified was returned."
		comment := fmt.Sprintf(commentFmtStr, retType, retType)
		p("%s", asComment("", comment))
	}
	pn("func (c *%s) Do() (%serror) {", callName, retTypeComma)
	nilRet := ""
	if retTypeComma != "" {
		nilRet = "nil, "
	}
	pn(`res, err := c.doRequest("json")`)
	if retTypeComma != "" && !mapRetType {
		pn("if res != nil && res.StatusCode == http.StatusNotModified {")
		pn(" if res.Body != nil { res.Body.Close() }")
		pn(" return nil, &googleapi.Error{")
		pn("  Code: res.StatusCode,")
		pn("  Header: res.Header,")
		pn(" }")
		pn("}")
	}
	pn("if err != nil { return %serr }", nilRet)
	pn("defer googleapi.CloseBody(res)")
	pn("if err := googleapi.CheckResponse(res); err != nil { return %serr }", nilRet)
	if meth.supportsMediaUpload() {
		pn(`if c.protocol_ == "resumable" {`)
		pn(` loc := res.Header.Get("Location")`)
		pn(" rx := &googleapi.ResumableUpload{")
		pn("  Client:        c.s.client,")
		pn("  UserAgent:     c.s.userAgent(),")
		pn("  URI:           loc,")
		pn("  Media:         c.resumable_,")
		pn("  MediaType:     c.mediaType_,")
		pn("  ContentLength: c.resumable_.Size(),")
		pn("  Callback:      func(curr int64){")
		pn("   if c.progressUpdater_ != nil {")
		pn("    c.progressUpdater_(curr, c.resumable_.Size())")
		pn("   }")
		pn("  },")
		pn(" }")
		pn(" res, err = rx.Upload(c.ctx_)")
		pn(" if err != nil { return %serr }", nilRet)
		pn(" defer res.Body.Close()")
		pn("}")
	}
	if retTypeComma == "" {
		pn("return nil")
	} else {
		if mapRetType {
			pn("var ret %s", responseType(a, meth.m))
		} else {
			pn("ret := &%s{", responseTypeLiteral(a, meth.m))
			pn(" ServerResponse: googleapi.ServerResponse{")
			pn("  Header: res.Header,")
			pn("  HTTPStatusCode: res.StatusCode,")
			pn(" },")
			pn("}")
		}
		pn("if err := json.NewDecoder(res.Body).Decode(&ret); err != nil { return nil, err }")
		pn("return ret, nil")
	}

	bs, _ := json.MarshalIndent(meth.m, "\t// ", "  ")
	pn("// %s\n", string(bs))
	pn("}")
}

// A Field provides methods that describe the characteristics of a Param or Property.
type Field interface {
	Default() string
	Enum() ([]string, bool)
	EnumDescriptions() []string
	UnfortunateDefault() bool
}

type Param struct {
	method        *Method
	name          string
	m             map[string]interface{}
	callFieldName string // empty means to use the default
}

func (p *Param) Default() string {
	return jstr(p.m, "default")
}

func (p *Param) Enum() ([]string, bool) {
	if e := jstrlist(p.m, "enum"); e != nil {
		return e, true
	}
	return nil, false
}

func (p *Param) EnumDescriptions() []string {
	return jstrlist(p.m, "enumDescriptions")
}

func (p *Param) UnfortunateDefault() bool {
	// We do not do anything special for Params with unfortunate defaults.
	return false
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

func (a *argument) exprAsString(prefix string) string {
	switch a.gotype {
	case "[]string":
		log.Printf("TODO(bradfitz): only including the first parameter in path query.")
		return prefix + a.goname + `[0]`
	case "string":
		return prefix + a.goname
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

var urlRE = regexp.MustCompile(`^http\S+$`)

func asComment(pfx, c string) string {
	var buf bytes.Buffer
	const maxLen = 70
	r := strings.NewReplacer(
		"\n", "\n"+pfx+"// ",
		"`\"", `"`,
		"\"`", `"`,
	)
	for len(c) > 0 {
		line := c
		if len(line) < maxLen {
			fmt.Fprintf(&buf, "%s// %s\n", pfx, r.Replace(line))
			break
		}
		// Don't break URLs.
		if !urlRE.MatchString(line[:maxLen]) {
			line = line[:maxLen]
		}
		si := strings.LastIndex(line, " ")
		if nl := strings.Index(line, "\n"); nl != -1 && nl < si {
			si = nl
		}
		if si != -1 {
			line = line[:si]
		}
		fmt.Fprintf(&buf, "%s// %s\n", pfx, r.Replace(line))
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
				return s.GoReturnType()
			}
			return "*" + ref
		}
	}
	return ""
}

// Strips the leading '*' from a type name so that it can be used to create a literal.
func responseTypeLiteral(api *API, m map[string]interface{}) string {
	v := responseType(api, m)
	if strings.HasPrefix(v, "*") {
		return v[1:]
	}
	return v
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

// depunct removes '-', '.', '$', '/', '_' from identifers, making the
// following character uppercase. Multiple '_' are preserved.
func depunct(ident string, needCap bool) string {
	var buf bytes.Buffer
	preserve_ := false
	for i, c := range ident {
		if c == '_' {
			if preserve_ || strings.HasPrefix(ident[i:], "__") {
				preserve_ = true
			} else {
				needCap = true
				continue
			}
		} else {
			preserve_ = false
		}
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

// jobj looks up the JSON object indexed by key in m.
func jobj(m map[string]interface{}, key string) map[string]interface{} {
	if m, ok := m[key].(map[string]interface{}); ok {
		return m
	}
	return nil
}

// jobj looks up the list of JSON objects indexed by key in m.
func jobjlist(m map[string]interface{}, key string) []map[string]interface{} {
	si, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	var sl []map[string]interface{}
	for _, si := range si {
		sl = append(sl, si.(map[string]interface{}))
	}
	return sl
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

func addFieldValueComments(p func(format string, args ...interface{}), field Field, indent string, blankLine bool) {
	var lines []string

	if enum, ok := field.Enum(); ok {
		desc := field.EnumDescriptions()
		lines = append(lines, asComment(indent, "Possible values:"))
		defval := field.Default()
		for i, v := range enum {
			more := ""
			if v == defval {
				more = " (default)"
			}
			if len(desc) > i && desc[i] != "" {
				more = more + " - " + desc[i]
			}
			lines = append(lines, asComment(indent, `  "`+v+`"`+more))
		}
	} else if field.UnfortunateDefault() {
		lines = append(lines, asComment("\t", fmt.Sprintf("Default: %s", field.Default())))
	}
	if blankLine && len(lines) > 0 {
		p(indent + "//\n")
	}
	for _, l := range lines {
		p("%s", l)
	}
}
