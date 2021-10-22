// Copyright 2011 Google LLC. All rights reserved.
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
	"time"
	"unicode"

	"google.golang.org/api/google-api-go-generator/internal/disco"
	"google.golang.org/api/internal/version"
)

const (
	googleDiscoveryURL = "https://www.googleapis.com/discovery/v1/apis"
)

var (
	apiToGenerate = flag.String("api", "*", "The API ID to generate, like 'tasks:v1'. A value of '*' means all.")
	useCache      = flag.Bool("cache", true, "Use cache of discovered Google API discovery documents.")
	genDir        = flag.String("gendir", defaultGenDir(), "Directory to use to write out generated Go files")
	build         = flag.Bool("build", false, "Compile generated packages.")
	install       = flag.Bool("install", false, "Install generated packages.")
	apisURL       = flag.String("discoveryurl", googleDiscoveryURL, "URL to root discovery document")

	publicOnly = flag.Bool("publiconly", true, "Only build public, released APIs. Only applicable for Google employees.")

	jsonFile       = flag.String("api_json_file", "", "If non-empty, the path to a local file on disk containing the API to generate. Exclusive with setting --api.")
	output         = flag.String("output", "", "(optional) Path to source output file. If not specified, the API name and version are used to construct an output path (e.g. tasks/v1).")
	apiPackageBase = flag.String("api_pkg_base", "google.golang.org/api", "Go package prefix to use for all generated APIs.")
	baseURL        = flag.String("base_url", "", "(optional) Override the default service API URL. If empty, the service's root URL will be used.")
	headerPath     = flag.String("header_path", "", "If non-empty, prepend the contents of this file to generated services.")

	gensupportPkg     = flag.String("gensupport_pkg", "google.golang.org/api/internal/gensupport", "Go package path of the 'api/internal/gensupport' support package.")
	googleapiPkg      = flag.String("googleapi_pkg", "google.golang.org/api/googleapi", "Go package path of the 'api/googleapi' support package.")
	optionPkg         = flag.String("option_pkg", "google.golang.org/api/option", "Go package path of the 'api/option' support package.")
	internalOptionPkg = flag.String("internaloption_pkg", "google.golang.org/api/option/internaloption", "Go package path of the 'api/option/internaloption' support package.")
	htransportPkg     = flag.String("htransport_pkg", "google.golang.org/api/transport/http", "Go package path of the 'api/transport/http' support package.")

	copyrightYear = flag.String("copyright_year", fmt.Sprintf("%d", time.Now().Year()), "Year for copyright.")

	serviceTypes = []string{"Service", "APIService"}
)

// API represents an API to generate, as well as its state while it's
// generating.
type API struct {
	// Fields needed before generating code, to select and find the APIs
	// to generate.
	// These fields usually come from the "directory item" JSON objects
	// that are provided by the googleDiscoveryURL. We unmarshal a directory
	// item directly into this struct.
	ID            string `json:"id"`
	Name          string `json:"name"`
	Version       string `json:"version"`
	DiscoveryLink string `json:"discoveryRestUrl"` // absolute

	doc *disco.Document
	// TODO(jba): remove m when we've fully converted to using disco.
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

func (a *API) Schema(name string) *Schema {
	return a.schemas[name]
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
		if err != nil && err != errNoDoc {
			errors = append(errors, &generateError{api, err})
			continue
		}
		if *build && err == nil {
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
			log.Println(ce.Error())
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
	var bytes []byte
	var source string
	apiListFile := filepath.Join(genDirRoot(), "api-list.json")
	if *useCache {
		if !*publicOnly {
			log.Fatalf("-cache=true not compatible with -publiconly=false")
		}
		var err error
		bytes, err = ioutil.ReadFile(apiListFile)
		if err != nil {
			log.Fatal(err)
		}
		source = apiListFile
	} else {
		bytes = slurpURL(*apisURL)
		if *publicOnly {
			if err := writeFile(apiListFile, bytes); err != nil {
				log.Fatal(err)
			}
		}
		source = *apisURL
	}
	apis, err := unmarshalAPIs(bytes)
	if err != nil {
		log.Fatalf("error decoding JSON in %s: %v", source, err)
	}
	if !*publicOnly && *apiToGenerate != "*" {
		apis = append(apis, apiFromID(*apiToGenerate))
	}
	return apis
}

func unmarshalAPIs(bytes []byte) ([]*API, error) {
	var itemObj struct{ Items []*API }
	if err := json.Unmarshal(bytes, &itemObj); err != nil {
		return nil, err
	}
	return itemObj.Items, nil
}

func apiFromID(apiID string) *API {
	parts := strings.Split(apiID, ":")
	if len(parts) != 2 {
		log.Fatalf("malformed API name: %q", apiID)
	}
	return &API{
		ID:      apiID,
		Name:    parts[0],
		Version: parts[1],
	}
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
	doc, err := disco.NewDocument(jsonBytes)
	if err != nil {
		return nil, fmt.Errorf("reading document from %q: %v", file, err)
	}
	a := &API{
		ID:        doc.ID,
		Name:      doc.Name,
		Version:   doc.Version,
		forceJSON: jsonBytes,
		doc:       doc,
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
	if res.StatusCode >= 300 {
		log.Printf("WARNING: URL %s served status code %d", urlStr, res.StatusCode)
		return nil
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
	if *genDir == "" {
		log.Fatalf("-gendir option must be set.")
	}
	return *genDir
}

func defaultGenDir() string {
	// TODO(cbro): consider using $CWD
	paths := filepath.SplitList(os.Getenv("GOPATH"))
	if len(paths) == 0 {
		return ""
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
	return a.DiscoveryLink
}

func (a *API) Package() string {
	return strings.ToLower(a.Name)
}

func (a *API) Target() string {
	return fmt.Sprintf("%s/%s/%s", *apiPackageBase, a.Package(), renameVersion(a.Version))
}

// ServiceType returns the name of the type to use for the root API struct
// (typically "Service").
func (a *API) ServiceType() string {
	if a.Name == "monitoring" && a.Version == "v3" {
		// HACK(deklerk) monitoring:v3 should always use call its overall
		// service struct "Service", even though there is a "Service" in its
		// schema (we re-map it to MService later).
		return "Service"
	}
	switch a.Name {
	case "appengine", "content": // retained for historical compatibility.
		return "APIService"
	default:
		for _, t := range serviceTypes {
			if _, ok := a.schemas[t]; !ok {
				return t
			}
		}
		panic("all service types are used, please consider introducing a new type to serviceTypes.")
	}
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
		base, rel = *baseURL, a.doc.BasePath
	case a.doc.RootURL != "":
		base, rel = a.doc.RootURL, a.doc.ServicePath
	default:
		base, rel = *apisURL, a.doc.BasePath
	}
	return resolveRelative(base, rel)
}

func (a *API) needsDataWrapper() bool {
	for _, feature := range a.doc.Features {
		if feature == "dataWrapper" {
			return true
		}
	}
	return false
}

func (a *API) jsonBytes() []byte {
	if a.forceJSON == nil {
		var slurp []byte
		var err error
		if *useCache {
			slurp, err = ioutil.ReadFile(a.JSONFile())
			if err != nil {
				log.Fatal(err)
			}
		} else {
			slurp = slurpURL(a.DiscoveryURL())
			if slurp != nil {
				// Make sure that keys are sorted by re-marshalling.
				d := make(map[string]interface{})
				json.Unmarshal(slurp, &d)
				if err != nil {
					log.Fatal(err)
				}
				var err error
				slurp, err = json.MarshalIndent(d, "", "  ")
				if err != nil {
					log.Fatal(err)
				}
			}
		}
		a.forceJSON = slurp
	}
	return a.forceJSON
}

func (a *API) JSONFile() string {
	return filepath.Join(a.SourceDir(), a.Package()+"-api.json")
}

var errNoDoc = errors.New("could not read discovery doc")

// WriteGeneratedCode generates code for a.
// It returns errNoDoc if we couldn't read the discovery doc.
func (a *API) WriteGeneratedCode() error {
	genfilename := *output
	jsonBytes := a.jsonBytes()
	// Skip generation if we don't have the discovery doc.
	if jsonBytes == nil {
		// No message here, because slurpURL printed one.
		return errNoDoc
	}
	if genfilename == "" {
		if err := writeFile(a.JSONFile(), jsonBytes); err != nil {
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
	if err != nil {
		return err
	}
	return nil
}

var docsLink string

func (a *API) GenerateCode() ([]byte, error) {
	pkg := a.Package()

	jsonBytes := a.jsonBytes()
	var err error
	if a.doc == nil {
		a.doc, err = disco.NewDocument(jsonBytes)
		if err != nil {
			return nil, err
		}
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

	if *headerPath != "" {
		if err := wf(*headerPath); err != nil {
			return nil, err
		}
	}

	pn(`// Copyright %s Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated file. DO NOT EDIT.
`, *copyrightYear)

	pn("// Package %s provides access to the %s.", pkg, a.doc.Title)
	if r := replacementPackage[pkg]; r != "" {
		pn("//")
		pn("// This package is DEPRECATED. Use package %s instead.", r)
	}
	docsLink = a.doc.DocumentationLink
	if docsLink != "" {
		pn("//")
		pn("// For product documentation, see: %s", docsLink)
	}
	pn("//")
	pn("// Creating a client")
	pn("//")
	pn("// Usage example:")
	pn("//")
	pn("//   import %q", a.Target())
	pn("//   ...")
	pn("//   ctx := context.Background()")
	pn("//   %sService, err := %s.NewService(ctx)", pkg, pkg)
	pn("//")
	pn("// In this example, Google Application Default Credentials are used for authentication.")
	pn("//")
	pn("// For information on how to create and obtain Application Default Credentials, see https://developers.google.com/identity/protocols/application-default-credentials.")
	pn("//")
	pn("// Other authentication options")
	pn("//")
	if len(a.doc.Auth.OAuth2Scopes) > 1 {
		pn(`// By default, all available scopes (see "Constants") are used to authenticate. To restrict scopes, use option.WithScopes:`)
		pn("//")
		// NOTE: the first scope tends to be the broadest. Use the last one to demonstrate restriction.
		pn("//   %sService, err := %s.NewService(ctx, option.WithScopes(%s.%s))", pkg, pkg, pkg, scopeIdentifier(a.doc.Auth.OAuth2Scopes[len(a.doc.Auth.OAuth2Scopes)-1]))
		pn("//")
	}
	pn("// To use an API key for authentication (note: some APIs do not support API keys), use option.WithAPIKey:")
	pn("//")
	pn(`//   %sService, err := %s.NewService(ctx, option.WithAPIKey("AIza..."))`, pkg, pkg)
	pn("//")
	pn("// To use an OAuth token (e.g., a user token obtained via a three-legged OAuth flow), use option.WithTokenSource:")
	pn("//")
	pn("//   config := &oauth2.Config{...}")
	pn("//   // ...")
	pn("//   token, err := config.Exchange(ctx, ...)")
	pn("//   %sService, err := %s.NewService(ctx, option.WithTokenSource(config.TokenSource(ctx, token)))", pkg, pkg)
	pn("//")
	pn("// See https://godoc.org/google.golang.org/api/option/ for details on options.")
	pn("package %s // import %q", pkg, a.Target())
	p("\n")
	pn("import (")
	for _, imp := range []string{
		"bytes",
		"context",
		"encoding/json",
		"errors",
		"fmt",
		"io",
		"net/http",
		"net/url",
		"strconv",
		"strings",
	} {
		pn("  %q", imp)
	}
	pn("")
	for _, imp := range []struct {
		pkg   string
		lname string
	}{
		{*gensupportPkg, "gensupport"},
		{*googleapiPkg, "googleapi"},
		{*optionPkg, "option"},
		{*internalOptionPkg, "internaloption"},
		{*htransportPkg, "htransport"},
	} {
		pn("  %s %q", imp.lname, imp.pkg)
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
	pn("var _ = internaloption.WithDefaultEndpoint")
	pn("")
	pn("const apiId = %q", a.doc.ID)
	pn("const apiName = %q", a.doc.Name)
	pn("const apiVersion = %q", a.doc.Version)
	pn("const basePath = %q", a.apiBaseURL())

	a.generateScopeConstants()
	a.PopulateSchemas()

	service := a.ServiceType()

	// Reserve names (ignore return value; we're the first caller).
	a.GetName("New")
	a.GetName(service)

	pn("// NewService creates a new %s.", service)
	pn("func NewService(ctx context.Context, opts ...option.ClientOption) (*%s, error) {", service)
	if len(a.doc.Auth.OAuth2Scopes) != 0 {
		pn("scopesOption := option.WithScopes(")
		for _, scope := range a.doc.Auth.OAuth2Scopes {
			pn("%q,", scope.ID)
		}
		pn(")")
		pn("// NOTE: prepend, so we don't override user-specified scopes.")
		pn("opts = append([]option.ClientOption{scopesOption}, opts...)")
	}
	pn("opts = append(opts, internaloption.WithDefaultEndpoint(basePath))")
	pn("client, endpoint, err := htransport.NewClient(ctx, opts...)")
	pn("if err != nil { return nil, err }")
	pn("s, err := New(client)")
	pn("if err != nil { return nil, err }")
	pn(`if endpoint != "" { s.BasePath = endpoint }`)
	pn("return s, nil")
	pn("}\n")

	pn("// New creates a new %s. It uses the provided http.Client for requests.", service)
	pn("//")
	pn("// Deprecated: please use NewService instead.")
	pn("// To provide a custom HTTP client, use option.WithHTTPClient.")
	pn("// If you are using google.golang.org/api/googleapis/transport.APIKey, use option.WithAPIKey with NewService instead.")
	pn("func New(client *http.Client) (*%s, error) {", service)
	pn("if client == nil { return nil, errors.New(\"client is nil\") }")
	pn("s := &%s{client: client, BasePath: basePath}", service)
	for _, res := range a.doc.Resources { // add top level resources.
		pn("s.%s = New%s(s)", resourceGoField(res, nil), resourceGoType(res))
	}
	pn("return s, nil")
	pn("}")

	pn("\ntype %s struct {", service)
	pn(" client *http.Client")
	pn(" BasePath string // API endpoint base URL")
	pn(" UserAgent string // optional additional User-Agent fragment")

	for _, res := range a.doc.Resources {
		pn("\n\t%s\t*%s", resourceGoField(res, nil), resourceGoType(res))
	}
	pn("}")
	pn("\nfunc (s *%s) userAgent() string {", service)
	pn(` if s.UserAgent == "" { return googleapi.UserAgent }`)
	pn(` return googleapi.UserAgent + " " + s.UserAgent`)
	pn("}\n")

	for _, res := range a.doc.Resources {
		a.generateResource(res)
	}

	a.responseTypes = make(map[string]bool)
	for _, meth := range a.APIMethods() {
		meth.cacheResponseTypes(a)
	}
	for _, res := range a.doc.Resources {
		a.cacheResourceResponseTypes(res)
	}

	for _, name := range a.sortedSchemaNames() {
		a.schemas[name].writeSchemaCode(a)
	}

	for _, meth := range a.APIMethods() {
		meth.generateCode()
	}

	for _, res := range a.doc.Resources {
		a.generateResourceMethods(res)
	}

	clean, err := format.Source(buf.Bytes())
	if err != nil {
		return buf.Bytes(), err
	}
	return clean, nil
}

func (a *API) generateScopeConstants() {
	scopes := a.doc.Auth.OAuth2Scopes
	if len(scopes) == 0 {
		return
	}

	a.pn("// OAuth2 scopes used by this API.")
	a.pn("const (")
	n := 0
	for _, scope := range scopes {
		if n > 0 {
			a.p("\n")
		}
		n++
		ident := scopeIdentifier(scope)
		if scope.Description != "" {
			a.p("%s", asComment("\t", scope.Description))
		}
		a.pn("\t%s = %q", ident, scope.ID)
	}
	a.p(")\n\n")
}

func scopeIdentifier(s disco.Scope) string {
	if s.ID == "openid" {
		return "OpenIDScope"
	}

	urlStr := s.ID
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

// Schema is a disco.Schema that has been bestowed an identifier, whether by
// having an "id" field at the top of the schema or with an
// automatically generated one in populateSubSchemas.
//
// TODO: While sub-types shouldn't need to be promoted to schemas,
// API.GenerateCode iterates over API.schemas to figure out what
// top-level Go types to write.  These should be separate concerns.
type Schema struct {
	api *API

	typ *disco.Schema

	apiName      string // the native API-defined name of this type
	goName       string // lazily populated by GoName
	goReturnType string // lazily populated by GoReturnType
	props        []*Property
}

type Property struct {
	s              *Schema // the containing Schema
	p              *disco.Property
	assignedGoName string
}

func (p *Property) Type() *disco.Schema {
	return p.p.Schema
}

func (p *Property) GoName() string {
	return initialCap(p.p.Name)
}

func (p *Property) Default() string {
	return p.p.Schema.Default
}

func (p *Property) Description() string {
	return p.p.Schema.Description
}

func (p *Property) Enum() ([]string, bool) {
	typ := p.p.Schema
	if typ.Enums != nil {
		return typ.Enums, true
	}
	// Check if this has an array of string enums.
	if typ.ItemSchema != nil {
		if enums := typ.ItemSchema.Enums; enums != nil && typ.ItemSchema.Type == "string" {
			return enums, true
		}
	}
	return nil, false
}

func (p *Property) EnumDescriptions() []string {
	if desc := p.p.Schema.EnumDescriptions; desc != nil {
		return desc
	}
	// Check if this has an array of string enum descriptions.
	if items := p.p.Schema.ItemSchema; items != nil {
		if desc := items.EnumDescriptions; desc != nil {
			return desc
		}
	}
	return nil
}

func (p *Property) Pattern() (string, bool) {
	return p.p.Schema.Pattern, (p.p.Schema.Pattern != "")
}

func (p *Property) TypeAsGo() string {
	return p.s.api.typeAsGo(p.Type(), false)
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
	{api: "androidpublisher:v1.1", schema: "InappPurchase", field: "PurchaseType"},
	{api: "androidpublisher:v2", schema: "ProductPurchase", field: "PurchaseType"},
	{api: "androidpublisher:v3", schema: "ProductPurchase", field: "PurchaseType"},
	{api: "androidpublisher:v2", schema: "SubscriptionPurchase", field: "CancelReason"},
	{api: "androidpublisher:v2", schema: "SubscriptionPurchase", field: "PaymentState"},
	{api: "androidpublisher:v2", schema: "SubscriptionPurchase", field: "PurchaseType"},
	{api: "androidpublisher:v3", schema: "SubscriptionPurchase", field: "PurchaseType"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "BoolValue"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "DoubleValue"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "Int64Value"},
	{api: "cloudmonitoring:v2beta2", schema: "Point", field: "StringValue"},
	{api: "compute:alpha", schema: "Scheduling", field: "AutomaticRestart"},
	{api: "compute:beta", schema: "MetadataItems", field: "Value"},
	{api: "compute:beta", schema: "Scheduling", field: "AutomaticRestart"},
	{api: "compute:v1", schema: "MetadataItems", field: "Value"},
	{api: "compute:v1", schema: "Scheduling", field: "AutomaticRestart"},
	{api: "content:v2", schema: "AccountUser", field: "Admin"},
	{api: "datastore:v1beta2", schema: "Property", field: "BlobKeyValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "BlobValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "BooleanValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "DateTimeValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "DoubleValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "Indexed"},
	{api: "datastore:v1beta2", schema: "Property", field: "IntegerValue"},
	{api: "datastore:v1beta2", schema: "Property", field: "StringValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "BlobValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "BooleanValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "DoubleValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "IntegerValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "StringValue"},
	{api: "datastore:v1beta3", schema: "Value", field: "TimestampValue"},
	{api: "genomics:v1beta2", schema: "Dataset", field: "IsPublic"},
	{api: "monitoring:v3", schema: "TypedValue", field: "BoolValue"},
	{api: "monitoring:v3", schema: "TypedValue", field: "DoubleValue"},
	{api: "monitoring:v3", schema: "TypedValue", field: "Int64Value"},
	{api: "monitoring:v3", schema: "TypedValue", field: "StringValue"},
	{api: "servicecontrol:v1", schema: "MetricValue", field: "BoolValue"},
	{api: "servicecontrol:v1", schema: "MetricValue", field: "DoubleValue"},
	{api: "servicecontrol:v1", schema: "MetricValue", field: "Int64Value"},
	{api: "servicecontrol:v1", schema: "MetricValue", field: "StringValue"},
	{api: "sqladmin:v1beta4", schema: "Settings", field: "StorageAutoResize"},
	{api: "storage:v1", schema: "BucketLifecycleRuleCondition", field: "IsLive"},
	{api: "storage:v1beta2", schema: "BucketLifecycleRuleCondition", field: "IsLive"},
	{api: "tasks:v1", schema: "Task", field: "Completed"},
	{api: "youtube:v3", schema: "ChannelSectionSnippet", field: "Position"},
	{api: "youtube:v3", schema: "MonitorStreamInfo", field: "EnableMonitorStream"},
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
	switch p.TypeAsGo() {
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

func (a *API) typeAsGo(s *disco.Schema, elidePointers bool) string {
	switch s.Kind {
	case disco.SimpleKind:
		return mustSimpleTypeConvert(s.Type, s.Format)
	case disco.ArrayKind:
		as := s.ElementSchema()
		if as.Type == "string" {
			switch as.Format {
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
			}
		}
		return "[]" + a.typeAsGo(as, elidePointers)
	case disco.ReferenceKind:
		rs := s.RefSchema
		if rs.Kind == disco.SimpleKind {
			// Simple top-level schemas get named types (see writeSchemaCode).
			// Use the name instead of using the equivalent simple Go type.
			return a.schemaNamed(rs.Name).GoName()
		}
		return a.typeAsGo(rs, elidePointers)
	case disco.MapKind:
		es := s.ElementSchema()
		if es.Type == "string" {
			// If the element schema has a type "string", it's going to be
			// transmitted as a string, and the Go map type must reflect that.
			// This is true even if the format is, say, "int64". When type =
			// "string" and format = "int64" at top level, we can use the json
			// "string" tag option to unmarshal the string to an int64, but
			// inside a map we can't.
			return "map[string]string"
		}
		// Due to historical baggage (maps used to be a separate code path),
		// the element types of maps never have pointers in them.  From this
		// level down, elide pointers in types.
		return "map[string]" + a.typeAsGo(es, true)
	case disco.AnyStructKind:
		return "googleapi.RawMessage"
	case disco.StructKind:
		tls := a.schemaNamed(s.Name)
		if elidePointers || s.Variant != nil {
			return tls.GoName()
		}
		return "*" + tls.GoName()
	default:
		panic(fmt.Sprintf("unhandled typeAsGo for %+v", s))
	}
}

func (a *API) schemaNamed(name string) *Schema {
	s := a.schemas[name]
	if s == nil {
		panicf("no top-level schema named %q", name)
	}
	return s
}

func (s *Schema) properties() []*Property {
	if s.props != nil {
		return s.props
	}
	if s.typ.Kind != disco.StructKind {
		panic("called properties on non-object schema")
	}
	for _, p := range s.typ.Properties {
		s.props = append(s.props, &Property{
			s: s,
			p: p,
		})
	}
	return s.props
}

func (s *Schema) HasContentType() bool {
	for _, p := range s.properties() {
		if p.GoName() == "ContentType" && p.TypeAsGo() == "string" {
			return true
		}
	}
	return false
}

func (s *Schema) populateSubSchemas() (outerr error) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		outerr = fmt.Errorf("%v", r)
	}()

	addSubStruct := func(subApiName string, t *disco.Schema) {
		if s.api.schemas[subApiName] != nil {
			panic("dup schema apiName: " + subApiName)
		}
		if t.Name != "" {
			panic("subtype already has name: " + t.Name)
		}
		t.Name = subApiName
		subs := &Schema{
			api:     s.api,
			typ:     t,
			apiName: subApiName,
		}
		s.api.schemas[subApiName] = subs
		err := subs.populateSubSchemas()
		if err != nil {
			panicf("in sub-struct %q: %v", subApiName, err)
		}
	}

	switch s.typ.Kind {
	case disco.StructKind:
		for _, p := range s.properties() {
			subApiName := fmt.Sprintf("%s.%s", s.apiName, p.p.Name)
			switch p.Type().Kind {
			case disco.SimpleKind, disco.ReferenceKind, disco.AnyStructKind:
				// Do nothing.
			case disco.MapKind:
				mt := p.Type().ElementSchema()
				if mt.Kind == disco.SimpleKind || mt.Kind == disco.ReferenceKind {
					continue
				}
				addSubStruct(subApiName, mt)
			case disco.ArrayKind:
				at := p.Type().ElementSchema()
				if at.Kind == disco.SimpleKind || at.Kind == disco.ReferenceKind {
					continue
				}
				addSubStruct(subApiName, at)
			case disco.StructKind:
				addSubStruct(subApiName, p.Type())
			default:
				panicf("Unknown type for %q: %v", subApiName, p.Type())
			}
		}
	case disco.ArrayKind:
		subApiName := fmt.Sprintf("%s.Item", s.apiName)
		switch at := s.typ.ElementSchema(); at.Kind {
		case disco.SimpleKind, disco.ReferenceKind, disco.AnyStructKind:
			// Do nothing.
		case disco.MapKind:
			mt := at.ElementSchema()
			if k := mt.Kind; k != disco.SimpleKind && k != disco.ReferenceKind {
				addSubStruct(subApiName, mt)
			}
		case disco.ArrayKind:
			at := at.ElementSchema()
			if k := at.Kind; k != disco.SimpleKind && k != disco.ReferenceKind {
				addSubStruct(subApiName, at)
			}
		case disco.StructKind:
			addSubStruct(subApiName, at)
		default:
			panicf("Unknown array type for %q: %v", subApiName, at)
		}
	case disco.AnyStructKind, disco.MapKind, disco.SimpleKind, disco.ReferenceKind:
		// Do nothing.
	default:
		fmt.Fprintf(os.Stderr, "in populateSubSchemas, schema is: %v", s.typ)
		panicf("populateSubSchemas: unsupported type for schema %q", s.apiName)
		panic("unreachable")
	}
	return nil
}

// GoName returns (or creates and returns) the bare Go name
// of the apiName, making sure that it's a proper Go identifier
// and doesn't conflict with an existing name.
func (s *Schema) GoName() string {
	if s.goName == "" {
		if s.typ.Kind == disco.MapKind {
			s.goName = s.api.typeAsGo(s.typ, false)
		} else {
			base := initialCap(s.apiName)

			// HACK(deklerk) Re-maps monitoring's Service field to MService so
			// that the overall struct for this API can keep its name "Service".
			// This takes care of "Service" the initial "goName" for "Service"
			// refs.
			if s.api.Name == "monitoring" && base == "Service" {
				base = "MService"
			}

			s.goName = s.api.GetName(base)
			if base == "Service" && s.goName != "Service" {
				// Detect the case where a resource is going to clash with the
				// root service object.
				panicf("Clash on name Service")
			}
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
		if s.typ.Kind == disco.MapKind {
			s.goReturnType = s.GoName()
		} else {
			s.goReturnType = "*" + s.GoName()
		}
	}
	return s.goReturnType
}

func (s *Schema) writeSchemaCode(api *API) {
	switch s.typ.Kind {
	case disco.SimpleKind:
		apitype := s.typ.Type
		typ := mustSimpleTypeConvert(apitype, s.typ.Format)
		s.api.pn("\ntype %s %s", s.GoName(), typ)
	case disco.StructKind:
		s.writeSchemaStruct(api)
	case disco.MapKind, disco.AnyStructKind:
		// Do nothing.
	case disco.ArrayKind:
		log.Printf("TODO writeSchemaCode for arrays for %s", s.GoName())
	default:
		fmt.Fprintf(os.Stderr, "in writeSchemaCode, schema is: %+v", s.typ)
		panicf("writeSchemaCode: unsupported type for schema %q", s.apiName)
	}
}

func (s *Schema) writeVariant(api *API, v *disco.Variant) {
	s.api.p("\ntype %s map[string]interface{}\n\n", s.GoName())

	// Write out the "Type" method that identifies the variant type.
	s.api.pn("func (t %s) Type() string {", s.GoName())
	s.api.pn("  return googleapi.VariantType(t)")
	s.api.p("}\n\n")

	// Write out helper methods to convert each possible variant.
	for _, m := range v.Map {
		if m.TypeValue == "" && m.Ref == "" {
			log.Printf("TODO variant %s ref %s not yet supported.", m.TypeValue, m.Ref)
			continue
		}

		s.api.pn("func (t %s) %s() (r %s, ok bool) {", s.GoName(), initialCap(m.TypeValue), m.Ref)
		s.api.pn(" if t.Type() != %q {", initialCap(m.TypeValue))
		s.api.pn("  return r, false")
		s.api.pn(" }")
		s.api.pn(" ok = googleapi.ConvertVariant(map[string]interface{}(t), &r)")
		s.api.pn(" return r, ok")
		s.api.p("}\n\n")
	}
}

func (s *Schema) Description() string {
	return s.typ.Description
}

func (s *Schema) writeSchemaStruct(api *API) {
	if v := s.typ.Variant; v != nil {
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
	nullFieldsName := np.Get("NullFields")
	if s.isResponseType() {
		np.Get("ServerResponse") // reserve the name
	}

	firstFieldName := "" // used to store a struct field name for use in documentation.
	for i, p := range s.properties() {
		if i > 0 {
			s.api.p("\n")
		}
		pname := np.Get(p.GoName())
		if pname[0] == '@' {
			// HACK(cbro): ignore JSON-LD special fields until we can figure out
			// the correct Go representation for them.
			continue
		}
		p.assignedGoName = pname
		des := p.Description()
		if des != "" {
			s.api.p("%s", asComment("\t", fmt.Sprintf("%s: %s", pname, des)))
		}
		addFieldValueComments(s.api.p, p, "\t", des != "")

		var extraOpt string
		if p.Type().IsIntAsString() {
			extraOpt += ",string"
		}

		typ := p.TypeAsGo()
		if p.forcePointerType() {
			typ = "*" + typ
		}

		s.api.pn(" %s %s `json:\"%s,omitempty%s\"`", pname, typ, p.p.Name, extraOpt)
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

	commentFmtStr = "%s is a list of field names (e.g. %q) to " +
		"include in API requests with the JSON null value. " +
		"By default, fields with empty values are omitted from API requests. However, " +
		"any field with an empty value appearing in %s will be sent to the server as null. " +
		"It is an error if a field in this list has a non-empty value. This may be used to " +
		"include null fields in Patch requests."
	comment = fmt.Sprintf(commentFmtStr, nullFieldsName, firstFieldName, nullFieldsName)
	s.api.p("\n")
	s.api.p("%s", asComment("\t", comment))

	s.api.pn("\t%s []string `json:\"-\"`", nullFieldsName)

	s.api.pn("}")
	s.writeSchemaMarshal(forceSendName, nullFieldsName)
	s.writeSchemaUnmarshal()
}

// writeSchemaMarshal writes a custom MarshalJSON function for s, which allows
// fields to be explicitly transmitted by listing them in the field identified
// by forceSendFieldName, and allows fields to be transmitted with the null value
// by listing them in the field identified by nullFieldsName.
func (s *Schema) writeSchemaMarshal(forceSendFieldName, nullFieldsName string) {
	s.api.pn("func (s *%s) MarshalJSON() ([]byte, error) {", s.GoName())
	s.api.pn("\ttype NoMethod %s", s.GoName())
	// pass schema as methodless type to prevent subsequent calls to MarshalJSON from recursing indefinitely.
	s.api.pn("\traw := NoMethod(*s)")
	s.api.pn("\treturn gensupport.MarshalJSON(raw, s.%s, s.%s)", forceSendFieldName, nullFieldsName)
	s.api.pn("}")
}

func (s *Schema) writeSchemaUnmarshal() {
	var floatProps []*Property
	for _, p := range s.properties() {
		if p.p.Schema.Type == "number" {
			floatProps = append(floatProps, p)
		}
	}
	if len(floatProps) == 0 {
		return
	}
	pn := s.api.pn
	pn("\nfunc (s *%s) UnmarshalJSON(data []byte) error {", s.GoName())
	pn("  type NoMethod %s", s.GoName()) // avoid infinite recursion
	pn("  var s1 struct {")
	// Hide the float64 fields of the schema with fields that correctly
	// unmarshal special values.
	for _, p := range floatProps {
		typ := "gensupport.JSONFloat64"
		if p.forcePointerType() {
			typ = "*" + typ
		}
		pn("%s %s `json:\"%s\"`", p.assignedGoName, typ, p.p.Name)
	}
	pn("    *NoMethod") // embed the schema
	pn("  }")
	// Set the schema value into the wrapper so its other fields are unmarshaled.
	pn("  s1.NoMethod = (*NoMethod)(s)")
	pn("  if err := json.Unmarshal(data, &s1); err != nil {")
	pn("    return err")
	pn("  }")
	// Copy each shadowing field into the field it shadows.
	for _, p := range floatProps {
		n := p.assignedGoName
		if p.forcePointerType() {
			pn("if s1.%s != nil { s.%s = (*float64)(s1.%s) }", n, n, n)
		} else {
			pn("s.%s = float64(s1.%s)", n, n)
		}
	}
	pn(" return nil")
	pn("}")
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
	if a.schemas != nil {
		panic("")
	}
	a.schemas = make(map[string]*Schema)
	for name, ds := range a.doc.Schemas {
		s := &Schema{
			api:     a,
			apiName: name,
			typ:     ds,
		}
		a.schemas[name] = s
		err := s.populateSubSchemas()
		if err != nil {
			panicf("Error populating schema with API name %q: %v", name, err)
		}
	}
}

func (a *API) generateResource(r *disco.Resource) {
	pn := a.pn
	t := resourceGoType(r)
	pn(fmt.Sprintf("func New%s(s *%s) *%s {", t, a.ServiceType(), t))
	pn("rs := &%s{s : s}", t)
	for _, res := range r.Resources {
		pn("rs.%s = New%s(s)", resourceGoField(res, r), resourceGoType(res))
	}
	pn("return rs")
	pn("}")

	pn("\ntype %s struct {", t)
	pn(" s *%s", a.ServiceType())
	for _, res := range r.Resources {
		pn("\n\t%s\t*%s", resourceGoField(res, r), resourceGoType(res))
	}
	pn("}")

	for _, res := range r.Resources {
		a.generateResource(res)
	}
}

func (a *API) cacheResourceResponseTypes(r *disco.Resource) {
	for _, meth := range a.resourceMethods(r) {
		meth.cacheResponseTypes(a)
	}
	for _, res := range r.Resources {
		a.cacheResourceResponseTypes(res)
	}
}

func (a *API) generateResourceMethods(r *disco.Resource) {
	for _, meth := range a.resourceMethods(r) {
		meth.generateCode()
	}
	for _, res := range r.Resources {
		a.generateResourceMethods(res)
	}
}

func resourceGoField(r, parent *disco.Resource) string {
	// Avoid conflicts with method names.
	und := ""
	if parent != nil {
		for _, m := range parent.Methods {
			if m.Name == r.Name {
				und = "_"
				break
			}
		}
	}
	// Note: initialCap(r.Name + "_") doesn't work because initialCap calls depunct.
	return initialCap(r.Name) + und
}

func resourceGoType(r *disco.Resource) string {
	return initialCap(r.FullName + "Service")
}

func (a *API) resourceMethods(r *disco.Resource) []*Method {
	ms := []*Method{}
	for _, m := range r.Methods {
		ms = append(ms, &Method{
			api: a,
			r:   r,
			m:   m,
		})
	}
	return ms
}

type Method struct {
	api *API
	r   *disco.Resource // or nil if a API-level (top-level) method
	m   *disco.Method

	params []*Param // all Params, of each type, lazily set by first call of Params method.
}

func (m *Method) Id() string {
	return m.m.ID
}

func (m *Method) responseType() *Schema {
	return m.api.schemas[m.m.Response.RefSchema.Name]
}

func (m *Method) supportsMediaUpload() bool {
	return m.m.MediaUpload != nil
}

func (m *Method) mediaUploadPath() string {
	return m.m.MediaUpload.Protocols["simple"].Path
}

func (m *Method) supportsMediaDownload() bool {
	if m.supportsMediaUpload() {
		// storage.objects.insert claims support for download in
		// addition to upload but attempting to do so fails.
		// This situation doesn't apply to any other methods.
		return false
	}
	return m.m.SupportsMediaDownload
}

func (m *Method) supportsPaging() (*pageTokenGenerator, string, bool) {
	ptg := m.pageTokenGenerator()
	if ptg == nil {
		return nil, "", false
	}

	// Check that the response type has the next page token.
	s := m.responseType()
	if s == nil || s.typ.Kind != disco.StructKind {
		return nil, "", false
	}
	for _, prop := range s.properties() {
		if isPageTokenName(prop.p.Name) && prop.Type().Type == "string" {
			return ptg, prop.GoName(), true
		}
	}

	return nil, "", false
}

type pageTokenGenerator struct {
	isParam     bool   // is the page token a URL parameter?
	name        string // param or request field name
	requestName string // empty for URL param
}

func (p *pageTokenGenerator) genGet() string {
	if p.isParam {
		return fmt.Sprintf("c.urlParams_.Get(%q)", p.name)
	}
	return fmt.Sprintf("c.%s.%s", p.requestName, p.name)
}

func (p *pageTokenGenerator) genSet(valueExpr string) string {
	if p.isParam {
		return fmt.Sprintf("c.%s(%s)", initialCap(p.name), valueExpr)
	}
	return fmt.Sprintf("c.%s.%s = %s", p.requestName, p.name, valueExpr)
}

func (p *pageTokenGenerator) genDeferBody() string {
	if p.isParam {
		return p.genSet(p.genGet())
	}
	return fmt.Sprintf("func (pt string) { %s }(%s)", p.genSet("pt"), p.genGet())
}

// pageTokenGenerator returns a pageTokenGenerator that will generate code to
// get/set the page token for a subsequent page in the context of the generated
// Pages method. It returns nil if there is no page token.
func (m *Method) pageTokenGenerator() *pageTokenGenerator {
	matches := m.grepParams(func(p *Param) bool { return isPageTokenName(p.p.Name) })
	switch len(matches) {
	case 1:
		if matches[0].p.Required {
			// The page token is a required parameter (e.g. because there is
			// a separate API call to start an iteration), and so the relevant
			// call factory method takes the page token instead.
			return nil
		}
		n := matches[0].p.Name
		return &pageTokenGenerator{true, n, ""}

	case 0: // No URL parameter, but maybe a request field.
		if m.m.Request == nil {
			return nil
		}
		rs := m.m.Request
		if rs.RefSchema != nil {
			rs = rs.RefSchema
		}
		for _, p := range rs.Properties {
			if isPageTokenName(p.Name) {
				return &pageTokenGenerator{false, initialCap(p.Name), validGoIdentifer(strings.ToLower(rs.Name))}
			}
		}
		return nil

	default:
		panicf("too many page token parameters for method %s", m.m.Name)
		return nil
	}
}

func isPageTokenName(s string) bool {
	return s == "pageToken" || s == "nextPageToken"
}

func (m *Method) Params() []*Param {
	if m.params == nil {
		for _, p := range m.m.Parameters {
			m.params = append(m.params, &Param{
				method: m,
				p:      p,
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
		return p.p.Name == name
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
		return !p.p.Required
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

func (meth *Method) generateCode() {
	res := meth.r // may be nil if a top-level method
	a := meth.api
	p, pn := a.p, a.pn

	pn("\n// method id %q:", meth.Id())

	retType := responseType(a, meth.m)
	if meth.IsRawResponse() {
		retType = "*http.Response"
	}
	retTypeComma := retType
	if retTypeComma != "" {
		retTypeComma += ", "
	}

	args := meth.NewArguments()
	methodName := initialCap(meth.m.Name)
	prefix := ""
	if res != nil {
		prefix = initialCap(res.FullName)
	}
	callName := a.GetName(prefix + methodName + "Call")

	pn("\ntype %s struct {", callName)
	pn(" s *%s", a.ServiceType())
	for _, arg := range args.l {
		if arg.location != "query" {
			pn(" %s %s", arg.goname, arg.gotype)
		}
	}
	pn(" urlParams_ gensupport.URLParams")
	httpMethod := meth.m.HTTPMethod
	if httpMethod == "GET" {
		pn(" ifNoneMatch_ string")
	}

	if meth.supportsMediaUpload() {
		pn(" mediaInfo_ *gensupport.MediaInfo")
	}
	pn(" ctx_ context.Context")
	pn(" header_ http.Header")
	pn("}")

	p("\n%s", asComment("", methodName+": "+meth.m.Description))
	if res != nil {
		if url := canonicalDocsURL[fmt.Sprintf("%v%v/%v", docsLink, res.Name, meth.m.Name)]; url != "" {
			pn("// For details, see %v", url)
		}
	}

	var servicePtr string
	if res == nil {
		pn("func (s *Service) %s(%s) *%s {", methodName, args, callName)
		servicePtr = "s"
	} else {
		pn("func (r *%s) %s(%s) *%s {", resourceGoType(res), methodName, args, callName)
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
		if opt.p.Location != "query" {
			panicf("optional parameter has unsupported location %q", opt.p.Location)
		}
		setter := initialCap(opt.p.Name)
		des := opt.p.Description
		des = strings.Replace(des, "Optional.", "", 1)
		des = strings.TrimSpace(des)
		p("\n%s", asComment("", fmt.Sprintf("%s sets the optional parameter %q: %s", setter, opt.p.Name, des)))
		addFieldValueComments(p, opt, "", true)
		np := new(namePool)
		np.Get("c") // take the receiver's name
		paramName := np.Get(validGoIdentifer(opt.p.Name))
		typePrefix := ""
		if opt.p.Repeated {
			typePrefix = "..."
		}
		pn("func (c *%s) %s(%s %s%s) *%s {", callName, setter, paramName, typePrefix, opt.GoType(), callName)
		if opt.p.Repeated {
			if opt.GoType() == "string" {
				pn("c.urlParams_.SetMulti(%q, append([]string{}, %v...))", opt.p.Name, paramName)
			} else {
				tmpVar := convertMultiParams(a, paramName)
				pn(" c.urlParams_.SetMulti(%q, %v)", opt.p.Name, tmpVar)
			}
		} else {
			if opt.GoType() == "string" {
				pn("c.urlParams_.Set(%q, %v)", opt.p.Name, paramName)
			} else {
				pn("c.urlParams_.Set(%q, fmt.Sprint(%v))", opt.p.Name, paramName)
			}
		}
		pn("return c")
		pn("}")
	}

	if meth.supportsMediaUpload() {
		comment := "Media specifies the media to upload in one or more chunks. " +
			"The chunk size may be controlled by supplying a MediaOption generated by googleapi.ChunkSize. " +
			"The chunk size defaults to googleapi.DefaultUploadChunkSize." +
			"The Content-Type header used in the upload request will be determined by sniffing the contents of r, " +
			"unless a MediaOption generated by googleapi.ContentType is supplied." +
			"\nAt most one of Media and ResumableMedia may be set."
		// TODO(mcgreevy): Ensure that r is always closed before Do returns, and document this.
		// See comments on https://code-review.googlesource.com/#/c/3970/
		p("\n%s", asComment("", comment))
		pn("func (c *%s) Media(r io.Reader, options ...googleapi.MediaOption) *%s {", callName, callName)
		// We check if the body arg, if any, has a content type and apply it here.
		// In practice, this only happens for the storage API today.
		// TODO(djd): check if we can cope with the developer setting the body's Content-Type field
		// after they've made this call.
		if ba := args.bodyArg(); ba != nil {
			if ba.schema.HasContentType() {
				pn("  if ct := c.%s.ContentType; ct != \"\" {", ba.goname)
				pn("   options = append([]googleapi.MediaOption{googleapi.ContentType(ct)}, options...)")
				pn("  }")
			}
		}
		pn(" c.mediaInfo_ = gensupport.NewInfoFromMedia(r, options)")
		pn(" return c")
		pn("}")
		comment = "ResumableMedia specifies the media to upload in chunks and can be canceled with ctx. " +
			"\n\nDeprecated: use Media instead." +
			"\n\nAt most one of Media and ResumableMedia may be set. " +
			`mediaType identifies the MIME media type of the upload, such as "image/png". ` +
			`If mediaType is "", it will be auto-detected. ` +
			`The provided ctx will supersede any context previously provided to ` +
			`the Context method.`
		p("\n%s", asComment("", comment))
		pn("func (c *%s) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *%s {", callName, callName)
		pn(" c.ctx_ = ctx")
		pn(" c.mediaInfo_ = gensupport.NewInfoFromResumableMedia(r, size, mediaType)")
		pn(" return c")
		pn("}")
		comment = "ProgressUpdater provides a callback function that will be called after every chunk. " +
			"It should be a low-latency function in order to not slow down the upload operation. " +
			"This should only be called when using ResumableMedia (as opposed to Media)."
		p("\n%s", asComment("", comment))
		pn("func (c *%s) ProgressUpdater(pu googleapi.ProgressUpdater) *%s {", callName, callName)
		pn(`c.mediaInfo_.SetProgressUpdater(pu)`)
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

	comment = "Header returns an http.Header that can be modified by the caller to add " +
		"HTTP headers to the request."
	p("\n%s", asComment("", comment))
	pn("func (c *%s) Header() http.Header {", callName)
	pn(" if c.header_ == nil {")
	pn("  c.header_ = make(http.Header)")
	pn(" }")
	pn(" return c.header_")
	pn("}")

	pn("\nfunc (c *%s) doRequest(alt string) (*http.Response, error) {", callName)
	pn(`reqHeaders := make(http.Header)`)
	pn(`reqHeaders.Set("x-goog-api-client", "gl-go/"+gensupport.GoVersion()+" gdcl/%s")`, version.Repo)
	pn("for k, v := range c.header_ {")
	pn(" reqHeaders[k] = v")
	pn("}")
	pn(`reqHeaders.Set("User-Agent",c.s.userAgent())`)
	if httpMethod == "GET" {
		pn(`if c.ifNoneMatch_ != "" {`)
		pn(` reqHeaders.Set("If-None-Match",  c.ifNoneMatch_)`)
		pn("}")
	}
	pn("var body io.Reader = nil")
	if meth.IsRawRequest() {
		pn("body = c.body_")
	} else {
		if ba := args.bodyArg(); ba != nil && httpMethod != "GET" {
			if meth.m.ID == "ml.projects.predict" {
				// TODO(cbro): move ML API to rawHTTP (it will be a breaking change)
				// Skip JSONReader for APIs that require clients to pass in JSON already.
				pn("body = strings.NewReader(c.%s.HttpBody.Data)", ba.goname)
			} else {
				style := "WithoutDataWrapper"
				if a.needsDataWrapper() {
					style = "WithDataWrapper"
				}
				pn("body, err := googleapi.%s.JSONReader(c.%s)", style, ba.goname)
				pn("if err != nil { return nil, err }")
			}

			pn(`reqHeaders.Set("Content-Type", "application/json")`)
		}
		pn(`c.urlParams_.Set("alt", alt)`)
		pn(`c.urlParams_.Set("prettyPrint", "false")`)
	}

	pn("urls := googleapi.ResolveRelative(c.s.BasePath, %q)", meth.m.Path)
	if meth.supportsMediaUpload() {
		pn("if c.mediaInfo_ != nil {")
		pn("  urls = googleapi.ResolveRelative(c.s.BasePath, %q)", meth.mediaUploadPath())
		pn(`  c.urlParams_.Set("uploadType", c.mediaInfo_.UploadType())`)
		pn("}")

		pn("if body == nil {")
		pn(" body = new(bytes.Buffer)")
		pn(` reqHeaders.Set("Content-Type", "application/json")`)
		pn("}")
		pn("body, getBody, cleanup := c.mediaInfo_.UploadRequest(reqHeaders, body)")
		pn("defer cleanup()")
	}
	pn(`urls += "?" + c.urlParams_.Encode()`)
	pn("req, err := http.NewRequest(%q, urls, body)", httpMethod)
	pn("if err != nil { return nil, err }")
	pn("req.Header = reqHeaders")
	if meth.supportsMediaUpload() {
		pn("req.GetBody = getBody")
	}

	// Replace param values after NewRequest to avoid reencoding them.
	// E.g. Cloud Storage API requires '%2F' in entity param to be kept, but url.Parse replaces it with '/'.
	argsForLocation := args.forLocation("path")
	if len(argsForLocation) > 0 {
		pn(`googleapi.Expand(req.URL, map[string]string{`)
		for _, arg := range argsForLocation {
			pn(`"%s": %s,`, arg.apiname, arg.exprAsString("c."))
		}
		pn(`})`)
	}

	pn("return gensupport.SendRequest(c.ctx_, c.s.client, req)")
	pn("}")

	if meth.supportsMediaDownload() {
		pn("\n// Download fetches the API endpoint's \"media\" value, instead of the normal")
		pn("// API response value. If the returned error is nil, the Response is guaranteed to")
		pn("// have a 2xx status code. Callers must close the Response.Body as usual.")
		pn("func (c *%s) Download(opts ...googleapi.CallOption) (*http.Response, error) {", callName)
		pn(`gensupport.SetOptions(c.urlParams_, opts...)`)
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
	pn("\n// Do executes the %q call.", meth.m.ID)
	if retTypeComma != "" && !mapRetType && !meth.IsRawResponse() {
		commentFmtStr := "Exactly one of %v or error will be non-nil. " +
			"Any non-2xx status code is an error. " +
			"Response headers are in either %v.ServerResponse.Header " +
			"or (if a response was returned at all) in error.(*googleapi.Error).Header. " +
			"Use googleapi.IsNotModified to check whether the returned error was because " +
			"http.StatusNotModified was returned."
		comment := fmt.Sprintf(commentFmtStr, retType, retType)
		p("%s", asComment("", comment))
	}
	pn("func (c *%s) Do(opts ...googleapi.CallOption) (%serror) {", callName, retTypeComma)
	nilRet := ""
	if retTypeComma != "" {
		nilRet = "nil, "
	}
	pn(`gensupport.SetOptions(c.urlParams_, opts...)`)
	if meth.IsRawResponse() {
		pn(`return c.doRequest("")`)
	} else {
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
			pn(`rx := c.mediaInfo_.ResumableUpload(res.Header.Get("Location"))`)
			pn("if rx != nil {")
			pn(" rx.Client = c.s.client")
			pn(" rx.UserAgent = c.s.userAgent()")
			pn(" ctx := c.ctx_")
			pn(" if ctx == nil {")
			// TODO(mcgreevy): Require context when calling Media, or Do.
			pn("  ctx = context.TODO()")
			pn(" }")
			pn(" res, err = rx.Upload(ctx)")
			pn(" if err != nil { return %serr }", nilRet)
			pn(" defer res.Body.Close()")
			pn(" if err := googleapi.CheckResponse(res); err != nil { return %serr }", nilRet)
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
			if a.needsDataWrapper() {
				pn("target := &struct {")
				pn("  Data %s `json:\"data\"`", responseType(a, meth.m))
				pn("}{ret}")
			} else {
				pn("target := &ret")
			}

			if meth.m.ID == "ml.projects.predict" {
				pn("var b bytes.Buffer")
				pn("if _, err := io.Copy(&b, res.Body); err != nil { return nil, err }")
				pn("if err := res.Body.Close(); err != nil { return nil, err }")
				pn("if err := json.NewDecoder(bytes.NewReader(b.Bytes())).Decode(target); err != nil { return nil, err }")
				pn("ret.Data = b.String()")
			} else {
				pn("if err := gensupport.DecodeResponse(target, res); err != nil { return nil, err }")
			}
			pn("return ret, nil")
		}
	}

	bs, err := json.MarshalIndent(meth.m.JSONMap, "\t// ", "  ")
	if err != nil {
		panic(err)
	}
	pn("// %s\n", string(bs))
	pn("}")

	if ptg, rname, ok := meth.supportsPaging(); ok {
		// We can assume retType is non-empty.
		pn("")
		pn("// Pages invokes f for each page of results.")
		pn("// A non-nil error returned from f will halt the iteration.")
		pn("// The provided context supersedes any context provided to the Context method.")
		pn("func (c *%s) Pages(ctx context.Context, f func(%s) error) error {", callName, retType)
		pn(" c.ctx_ = ctx")
		pn(` defer %s  // reset paging to original point`, ptg.genDeferBody())
		pn(" for {")
		pn("  x, err := c.Do()")
		pn("  if err != nil { return err }")
		pn("  if err := f(x); err != nil { return err }")
		pn(`  if x.%s == "" { return nil }`, rname)
		pn(ptg.genSet("x." + rname))
		pn(" }")
		pn("}")
	}
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
	p             *disco.Parameter
	callFieldName string // empty means to use the default
}

func (p *Param) Default() string {
	return p.p.Default
}

func (p *Param) Enum() ([]string, bool) {
	if e := p.p.Enums; e != nil {
		return e, true
	}
	return nil, false
}

func (p *Param) EnumDescriptions() []string {
	return p.p.EnumDescriptions
}

func (p *Param) UnfortunateDefault() bool {
	// We do not do anything special for Params with unfortunate defaults.
	return false
}

func (p *Param) GoType() string {
	typ, format := p.p.Type, p.p.Format
	if typ == "string" && strings.Contains(format, "int") && p.p.Location != "query" {
		panic("unexpected int parameter encoded as string, not in query: " + p.p.Name)
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
	return validGoIdentifer(p.p.Name)
}

// APIMethods returns top-level ("API-level") methods. They don't have an associated resource.
func (a *API) APIMethods() []*Method {
	meths := []*Method{}
	for _, m := range a.doc.Methods {
		meths = append(meths, &Method{
			api: a,
			r:   nil, // to be explicit
			m:   m,
		})
	}
	return meths
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

func (meth *Method) IsRawRequest() bool {
	if meth.m.Request == nil {
		return false
	}
	// TODO(cbro): enable across other APIs.
	if meth.api.Name != "healthcare" {
		return false
	}
	return meth.m.Request.Ref == "HttpBody"
}

func (meth *Method) IsRawResponse() bool {
	if meth.m.Response == nil {
		return false
	}
	if meth.IsRawRequest() {
		// always match raw requests with raw responses.
		return true
	}
	// TODO(cbro): enable across other APIs.
	if meth.api.Name != "healthcare" {
		return false
	}
	return meth.m.Response.Ref == "HttpBody"
}

func (meth *Method) NewArguments() *arguments {
	args := &arguments{
		method: meth,
		m:      make(map[string]*argument),
	}
	pnames := meth.m.ParameterOrder
	if len(pnames) == 0 {
		// No parameterOrder; collect required parameters and sort by name.
		for _, reqParam := range meth.grepParams(func(p *Param) bool { return p.p.Required }) {
			pnames = append(pnames, reqParam.p.Name)
		}
		sort.Strings(pnames)
	}
	for _, pname := range pnames {
		arg := meth.NewArg(pname, meth.NamedParam(pname))
		args.AddArg(arg)
	}
	if rs := meth.m.Request; rs != nil {
		if meth.IsRawRequest() {
			args.AddArg(&argument{
				goname: "body_",
				gotype: "io.Reader",
			})
		} else {
			args.AddArg(meth.NewBodyArg(rs))
		}
	}
	return args
}

func (meth *Method) NewBodyArg(ds *disco.Schema) *argument {
	s := meth.api.schemaNamed(ds.RefSchema.Name)
	return &argument{
		goname:   validGoIdentifer(strings.ToLower(ds.Ref)),
		apiname:  "REQUEST",
		gotype:   "*" + s.GoName(),
		apitype:  ds.Ref,
		location: "body",
		schema:   s,
	}
}

func (meth *Method) NewArg(apiname string, p *Param) *argument {
	apitype := p.p.Type
	des := p.p.Description
	goname := validGoIdentifer(apiname) // but might be changed later, if conflicts
	if strings.Contains(des, "identifier") && !strings.HasSuffix(strings.ToLower(goname), "id") {
		goname += "id" // yay
		p.callFieldName = goname
	}
	gotype := mustSimpleTypeConvert(apitype, p.p.Format)
	if p.p.Repeated {
		gotype = "[]" + gotype
	}
	return &argument{
		apiname:  apiname,
		apitype:  apitype,
		goname:   goname,
		gotype:   gotype,
		location: p.p.Location,
	}
}

type argument struct {
	method           *Method
	schema           *Schema // Set if location == "body".
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
	case "bool":
		return "strconv.FormatBool(" + prefix + a.goname + ")"
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

func responseType(api *API, m *disco.Method) string {
	if m.Response == nil {
		return ""
	}
	ref := m.Response.Ref
	if ref != "" {
		if s := api.schemas[ref]; s != nil {
			return s.GoReturnType()
		}
		return "*" + ref
	}
	return ""
}

// Strips the leading '*' from a type name so that it can be used to create a literal.
func responseTypeLiteral(api *API, m *disco.Method) string {
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
