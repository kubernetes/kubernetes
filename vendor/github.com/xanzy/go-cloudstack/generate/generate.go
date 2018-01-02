//
// Copyright 2016, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"sort"
	"strings"
	"unicode"
)

type apiInfo map[string][]string

const pkg = "cloudstack"

type allServices struct {
	services services
}

type apiInfoNotFoundError struct {
	api string
}

func (e *apiInfoNotFoundError) Error() string {
	return fmt.Sprintf("Could not find API details for: %s", e.api)
}

type generateError struct {
	service *service
	error   error
}

func (e *generateError) Error() string {
	return fmt.Sprintf("API %s failed to generate code: %v", e.service.name, e.error)
}

type goimportError struct {
	output string
}

func (e *goimportError) Error() string {
	return fmt.Sprintf("GoImport failed to format:\n%v", e.output)
}

type service struct {
	name string
	apis []*API

	p  func(format string, args ...interface{}) // print raw
	pn func(format string, args ...interface{}) // print with indent and newline
}

type services []*service

// Add functions for the Sort interface
func (s services) Len() int {
	return len(s)
}

func (s services) Less(i, j int) bool {
	return s[i].name < s[j].name
}

func (s services) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// APIParams represents a list of API params
type APIParams []*APIParam

// Add functions for the Sort interface
func (s APIParams) Len() int {
	return len(s)
}

func (s APIParams) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

func (s APIParams) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// API represents an API endpoint we can call
type API struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Isasync     bool         `json:"isasync"`
	Params      APIParams    `json:"params"`
	Response    APIResponses `json:"response"`
}

// APIParam represents a single API parameter
type APIParam struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Type        string `json:"type"`
	Required    bool   `json:"required"`
}

// APIResponse represents a API response
type APIResponse struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Type        string       `json:"type"`
	Response    APIResponses `json:"response,omitempty"`
}

// APIResponses represents a list of API responses
type APIResponses []*APIResponse

// Add functions for the Sort interface
func (s APIResponses) Len() int {
	return len(s)
}

func (s APIResponses) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

func (s APIResponses) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func main() {
	as, errors, err := getAllServices()
	if err != nil {
		log.Fatal(err)
	}

	if err := as.WriteGeneralCode(); err != nil {
		log.Fatal(err)
	}

	for _, s := range as.services {
		if err := s.WriteGeneratedCode(); err != nil {
			errors = append(errors, &generateError{s, err})
		}
	}

	outdir, err := sourceDir()
	if err != nil {
		log.Fatal(err)
	}
	out, err := exec.Command("goimports", "-w", outdir).CombinedOutput()
	if err != nil {
		errors = append(errors, &goimportError{string(out)})
	}

	if len(errors) > 0 {
		log.Printf("%d API(s) failed to generate:", len(errors))
		for _, ce := range errors {
			log.Printf(ce.Error())
		}
		os.Exit(1)
	}
}

func (as *allServices) WriteGeneralCode() error {
	outdir, err := sourceDir()
	if err != nil {
		log.Fatalf("Failed to get source dir: %s", err)
	}

	code, err := as.GeneralCode()
	if err != nil {
		return err
	}

	file := path.Join(outdir, "cloudstack.go")
	return ioutil.WriteFile(file, code, 0644)
}

func (as *allServices) GeneralCode() ([]byte, error) {
	// Buffer the output in memory, for gofmt'ing later in the defer.
	var buf bytes.Buffer
	p := func(format string, args ...interface{}) {
		_, err := fmt.Fprintf(&buf, format, args...)
		if err != nil {
			panic(err)
		}
	}
	pn := func(format string, args ...interface{}) {
		p(format+"\n", args...)
	}
	pn("//")
	pn("// Copyright 2016, Sander van Harmelen")
	pn("//")
	pn("// Licensed under the Apache License, Version 2.0 (the \"License\");")
	pn("// you may not use this file except in compliance with the License.")
	pn("// You may obtain a copy of the License at")
	pn("//")
	pn("//     http://www.apache.org/licenses/LICENSE-2.0")
	pn("//")
	pn("// Unless required by applicable law or agreed to in writing, software")
	pn("// distributed under the License is distributed on an \"AS IS\" BASIS,")
	pn("// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.")
	pn("// See the License for the specific language governing permissions and")
	pn("// limitations under the License.")
	pn("//")
	pn("")
	pn("package %s", pkg)
	pn("")
	pn("// UnlimitedResourceID is a special ID to define an unlimited resource")
	pn("const UnlimitedResourceID = \"-1\"")
	pn("")
	pn("var idRegex = regexp.MustCompile(`^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|-1)$`)")
	pn("")
	pn("// IsID return true if the passed ID is either a UUID or a UnlimitedResourceID")
	pn("func IsID(id string) bool {")
	pn("	return idRegex.MatchString(id)")
	pn("}")
	pn("")
	pn("// OptionFunc can be passed to the courtesy helper functions to set additional parameters")
	pn("type OptionFunc func(*CloudStackClient, interface{}) error")
	pn("")
	pn("type CSError struct {")
	pn("	ErrorCode   int    `json:\"errorcode\"`")
	pn("	CSErrorCode int    `json:\"cserrorcode\"`")
	pn("	ErrorText   string `json:\"errortext\"`")
	pn("}")
	pn("")
	pn("func (e *CSError) Error() error {")
	pn("	return fmt.Errorf(\"CloudStack API error %%d (CSExceptionErrorCode: %%d): %%s\", e.ErrorCode, e.CSErrorCode, e.ErrorText)")
	pn("}")
	pn("")
	pn("type CloudStackClient struct {")
	pn("	HTTPGETOnly bool // If `true` only use HTTP GET calls")
	pn("")
	pn("	client  *http.Client // The http client for communicating")
	pn("	baseURL string       // The base URL of the API")
	pn("	apiKey  string       // Api key")
	pn("	secret  string       // Secret key")
	pn("	async   bool         // Wait for async calls to finish")
	pn("	timeout int64        // Max waiting timeout in seconds for async jobs to finish; defaults to 300 seconds")
	pn("")
	for _, s := range as.services {
		pn("  %s *%s", strings.TrimSuffix(s.name, "Service"), s.name)
	}
	pn("}")
	pn("")
	pn("// Creates a new client for communicating with CloudStack")
	pn("func newClient(apiurl string, apikey string, secret string, async bool, verifyssl bool) *CloudStackClient {")
	pn("	cs := &CloudStackClient{")
	pn("		client: &http.Client{")
	pn("			Transport: &http.Transport{")
	pn("				Proxy:           http.ProxyFromEnvironment,")
	pn("				TLSClientConfig: &tls.Config{InsecureSkipVerify: !verifyssl}, // If verifyssl is true, skipping the verify should be false and vice versa")
	pn("			},")
	pn("		Timeout: time.Duration(60 * time.Second),")
	pn("		},")
	pn("		baseURL: apiurl,")
	pn("		apiKey:  apikey,")
	pn("		secret:  secret,")
	pn("		async:   async,")
	pn("		timeout: 300,")
	pn("	}")
	for _, s := range as.services {
		pn("	cs.%s = New%s(cs)", strings.TrimSuffix(s.name, "Service"), s.name)
	}
	pn("	return cs")
	pn("}")
	pn("")
	pn("// Default non-async client. So for async calls you need to implement and check the async job result yourself. When using")
	pn("// HTTPS with a self-signed certificate to connect to your CloudStack API, you would probably want to set 'verifyssl' to")
	pn("// false so the call ignores the SSL errors/warnings.")
	pn("func NewClient(apiurl string, apikey string, secret string, verifyssl bool) *CloudStackClient {")
	pn("	cs := newClient(apiurl, apikey, secret, false, verifyssl)")
	pn("	return cs")
	pn("}")
	pn("")
	pn("// For sync API calls this client behaves exactly the same as a standard client call, but for async API calls")
	pn("// this client will wait until the async job is finished or until the configured AsyncTimeout is reached. When the async")
	pn("// job finishes successfully it will return actual object received from the API and nil, but when the timout is")
	pn("// reached it will return the initial object containing the async job ID for the running job and a warning.")
	pn("func NewAsyncClient(apiurl string, apikey string, secret string, verifyssl bool) *CloudStackClient {")
	pn("	cs := newClient(apiurl, apikey, secret, true, verifyssl)")
	pn("	return cs")
	pn("}")
	pn("")
	pn("// When using the async client an api call will wait for the async call to finish before returning. The default is to poll for 300 seconds")
	pn("// seconds, to check if the async job is finished.")
	pn("func (cs *CloudStackClient) AsyncTimeout(timeoutInSeconds int64) {")
	pn("	cs.timeout = timeoutInSeconds")
	pn("}")
	pn("")
	pn("var AsyncTimeoutErr = errors.New(\"Timeout while waiting for async job to finish\")")
	pn("")
	pn("// A helper function that you can use to get the result of a running async job. If the job is not finished within the configured")
	pn("// timeout, the async job returns a AsyncTimeoutErr.")
	pn("func (cs *CloudStackClient) GetAsyncJobResult(jobid string, timeout int64) (json.RawMessage, error) {")
	pn("	var timer time.Duration")
	pn("	currentTime := time.Now().Unix()")
	pn("")
	pn("		for {")
	pn("		p := cs.Asyncjob.NewQueryAsyncJobResultParams(jobid)")
	pn("		r, err := cs.Asyncjob.QueryAsyncJobResult(p)")
	pn("		if err != nil {")
	pn("			return nil, err")
	pn("		}")
	pn("")
	pn("		// Status 1 means the job is finished successfully")
	pn("		if r.Jobstatus == 1 {")
	pn("			return r.Jobresult, nil")
	pn("		}")
	pn("")
	pn("		// When the status is 2, the job has failed")
	pn("		if r.Jobstatus == 2 {")
	pn("			if r.Jobresulttype == \"text\" {")
	pn("				return nil, fmt.Errorf(string(r.Jobresult))")
	pn("			} else {")
	pn("				return nil, fmt.Errorf(\"Undefined error: %%s\", string(r.Jobresult))")
	pn("			}")
	pn("		}")
	pn("")
	pn("		if time.Now().Unix()-currentTime > timeout {")
	pn("			return nil, AsyncTimeoutErr")
	pn("		}")
	pn("")
	pn("		// Add an (extremely simple) exponential backoff like feature to prevent")
	pn("		// flooding the CloudStack API")
	pn("		if timer < 15 {")
	pn("			timer++")
	pn("		}")
	pn("")
	pn("		time.Sleep(timer * time.Second)")
	pn("	}")
	pn("}")
	pn("")
	pn("// Execute the request against a CS API. Will return the raw JSON data returned by the API and nil if")
	pn("// no error occured. If the API returns an error the result will be nil and the HTTP error code and CS")
	pn("// error details. If a processing (code) error occurs the result will be nil and the generated error")
	pn("func (cs *CloudStackClient) newRequest(api string, params url.Values) (json.RawMessage, error) {")
	pn("	params.Set(\"apiKey\", cs.apiKey)")
	pn("	params.Set(\"command\", api)")
	pn("	params.Set(\"response\", \"json\")")
	pn("")
	pn("	// Generate signature for API call")
	pn("	// * Serialize parameters, URL encoding only values and sort them by key, done by encodeValues")
	pn("	// * Convert the entire argument string to lowercase")
	pn("	// * Replace all instances of '+' to '%%20'")
	pn("	// * Calculate HMAC SHA1 of argument string with CloudStack secret")
	pn("	// * URL encode the string and convert to base64")
	pn("	s := encodeValues(params)")
	pn("	s2 := strings.ToLower(s)")
	pn("	s3 := strings.Replace(s2, \"+\", \"%%20\", -1)")
	pn("	mac := hmac.New(sha1.New, []byte(cs.secret))")
	pn("	mac.Write([]byte(s3))")
	pn("	signature := base64.StdEncoding.EncodeToString(mac.Sum(nil))")
	pn("")
	pn("	var err error")
	pn("	var resp *http.Response")
	pn("	if !cs.HTTPGETOnly && (api == \"deployVirtualMachine\" || api == \"updateVirtualMachine\") {")
	pn("		// The deployVirtualMachine API should be called using a POST call")
	pn("  	// so we don't have to worry about the userdata size")
	pn("")
	pn("		// Add the unescaped signature to the POST params")
	pn("		params.Set(\"signature\", signature)")
	pn("")
	pn("		// Make a POST call")
	pn("		resp, err = cs.client.PostForm(cs.baseURL, params)")
	pn("	} else {")
	pn("		// Create the final URL before we issue the request")
	pn("		url := cs.baseURL + \"?\" + s + \"&signature=\" + url.QueryEscape(signature)")
	pn("")
	pn("		// Make a GET call")
	pn("		resp, err = cs.client.Get(url)")
	pn("	}")
	pn("	if err != nil {")
	pn("		return nil, err")
	pn("	}")
	pn("	defer resp.Body.Close()")
	pn("")
	pn("	b, err := ioutil.ReadAll(resp.Body)")
	pn("	if err != nil {")
	pn("		return nil, err")
	pn("	}")
	pn("")
	pn("	// Need to get the raw value to make the result play nice")
	pn("	b, err = getRawValue(b)")
	pn("	if err != nil {")
	pn("		return nil, err")
	pn("	}")
	pn("")
	pn("	if resp.StatusCode != 200 {")
	pn("		var e CSError")
	pn("		if err := json.Unmarshal(b, &e); err != nil {")
	pn("			return nil, err")
	pn("		}")
	pn("		return nil, e.Error()")
	pn("	}")
	pn("	return b, nil")
	pn("}")
	pn("")
	pn("// Custom version of net/url Encode that only URL escapes values")
	pn("// Unmodified portions here remain under BSD license of The Go Authors: https://go.googlesource.com/go/+/master/LICENSE")
	pn("func encodeValues(v url.Values) string {")
	pn("	if v == nil {")
	pn("		return \"\"")
	pn("	}")
	pn("	var buf bytes.Buffer")
	pn("	keys := make([]string, 0, len(v))")
	pn("	for k := range v {")
	pn("		keys = append(keys, k)")
	pn("	}")
	pn("	sort.Strings(keys)")
	pn("	for _, k := range keys {")
	pn("		vs := v[k]")
	pn("		prefix := k + \"=\"")
	pn("		for _, v := range vs {")
	pn("			if buf.Len() > 0 {")
	pn("				buf.WriteByte('&')")
	pn("			}")
	pn("			buf.WriteString(prefix)")
	pn("			buf.WriteString(url.QueryEscape(v))")
	pn("		}")
	pn("	}")
	pn("	return buf.String()")
	pn("}")
	pn("")
	pn("// Generic function to get the first raw value from a response as json.RawMessage")
	pn("func getRawValue(b json.RawMessage) (json.RawMessage, error) {")
	pn("	var m map[string]json.RawMessage")
	pn("	if err := json.Unmarshal(b, &m); err != nil {")
	pn("		return nil, err")
	pn("	}")
	pn("	for _, v := range m {")
	pn("		return v, nil")
	pn("	}")
	pn("	return nil, fmt.Errorf(\"Unable to extract the raw value from:\\n\\n%%s\\n\\n\", string(b))")
	pn("}")
	pn("")
	pn("// ProjectIDSetter is an interface that every type that can set a project ID must implement")
	pn("type ProjectIDSetter interface {")
	pn("	SetProjectid(string)")
	pn("}")
	pn("")
	pn("// WithProject takes either a project name or ID and sets the `projectid` parameter")
	pn("func WithProject(project string) OptionFunc {")
	pn("	return func(cs *CloudStackClient, p interface{}) error {")
	pn("		ps, ok := p.(ProjectIDSetter)")
	pn("")
	pn("		if !ok || project == \"\" {")
	pn("			return nil")
	pn("		}")
	pn("")
	pn("		if !IsID(project) {")
	pn("			id, _, err := cs.Project.GetProjectID(project)")
	pn("			if err != nil {")
	pn("				return err")
	pn("			}")
	pn("			project = id")
	pn("		}")
	pn("")
	pn("		ps.SetProjectid(project)")
	pn("")
	pn("		return nil")
	pn("	}")
	pn("}")
	pn("")
	pn("// VPCIDSetter is an interface that every type that can set a vpc ID must implement")
	pn("type VPCIDSetter interface {")
	pn("	SetVpcid(string)")
	pn("}")
	pn("")
	pn("// WithVPCID takes a vpc ID and sets the `vpcid` parameter")
	pn("func WithVPCID(id string) OptionFunc {")
	pn("	return func(cs *CloudStackClient, p interface{}) error {")
	pn("		vs, ok := p.(VPCIDSetter)")
	pn("")
	pn("		if !ok || id == \"\" {")
	pn("			return nil")
	pn("		}")
	pn("")
	pn("		vs.SetVpcid(id)")
	pn("")
	pn("		return nil")
	pn("	}")
	pn("}")
	pn("")
	for _, s := range as.services {
		pn("type %s struct {", s.name)
		pn("  cs *CloudStackClient")
		pn("}")
		pn("")
		pn("func New%s(cs *CloudStackClient) *%s {", s.name, s.name)
		pn("	return &%s{cs: cs}", s.name)
		pn("}")
		pn("")
	}

	clean, err := format.Source(buf.Bytes())
	if err != nil {
		return buf.Bytes(), err
	}
	return clean, err
}

func (s *service) WriteGeneratedCode() error {
	outdir, err := sourceDir()
	if err != nil {
		log.Fatalf("Failed to get source dir: %s", err)
	}

	code, err := s.GenerateCode()
	if err != nil {
		return err
	}

	file := path.Join(outdir, s.name+".go")
	return ioutil.WriteFile(file, code, 0644)
}

func (s *service) GenerateCode() ([]byte, error) {
	// Buffer the output in memory, for gofmt'ing later in the defer.
	var buf bytes.Buffer
	s.p = func(format string, args ...interface{}) {
		_, err := fmt.Fprintf(&buf, format, args...)
		if err != nil {
			panic(err)
		}
	}
	s.pn = func(format string, args ...interface{}) {
		s.p(format+"\n", args...)
	}
	pn := s.pn

	pn("//")
	pn("// Copyright 2016, Sander van Harmelen")
	pn("//")
	pn("// Licensed under the Apache License, Version 2.0 (the \"License\");")
	pn("// you may not use this file except in compliance with the License.")
	pn("// You may obtain a copy of the License at")
	pn("//")
	pn("//     http://www.apache.org/licenses/LICENSE-2.0")
	pn("//")
	pn("// Unless required by applicable law or agreed to in writing, software")
	pn("// distributed under the License is distributed on an \"AS IS\" BASIS,")
	pn("// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.")
	pn("// See the License for the specific language governing permissions and")
	pn("// limitations under the License.")
	pn("//")
	pn("")
	pn("package %s", pkg)
	pn("")
	if s.name == "FirewallService" {
		pn("// Helper function for maintaining backwards compatibility")
		pn("func convertFirewallServiceResponse(b []byte) ([]byte, error) {")
		pn("	var raw map[string]interface{}")
		pn("	if err := json.Unmarshal(b, &raw); err != nil {")
		pn("		return nil, err")
		pn("	}")
		pn("")
		pn("	if _, ok := raw[\"firewallrule\"]; ok {")
		pn("		return convertFirewallServiceListResponse(b)")
		pn("	}")
		pn("")
		pn("	for _, k := range []string{\"endport\", \"startport\"} {")
		pn("		if sVal, ok := raw[k].(string); ok {")
		pn("			iVal, err := strconv.Atoi(sVal)")
		pn("			if err != nil {")
		pn("				return nil, err")
		pn("			}")
		pn("			raw[k] = iVal")
		pn("		}")
		pn("	}")
		pn("")
		pn("	return json.Marshal(raw)")
		pn("}")
		pn("")
		pn("// Helper function for maintaining backwards compatibility")
		pn("func convertFirewallServiceListResponse(b []byte) ([]byte, error) {")
		pn("	var rawList struct {")
		pn("		Count         int                      `json:\"count\"`")
		pn("		FirewallRules []map[string]interface{} `json:\"firewallrule\"`")
		pn("	}")
		pn("")
		pn("	if err := json.Unmarshal(b, &rawList); err != nil {")
		pn("		return nil, err")
		pn("	}")
		pn("")
		pn("	for _, r := range rawList.FirewallRules {")
		pn("		for _, k := range []string{\"endport\", \"startport\"} {")
		pn("			if sVal, ok := r[k].(string); ok {")
		pn("				iVal, err := strconv.Atoi(sVal)")
		pn("				if err != nil {")
		pn("					return nil, err")
		pn("				}")
		pn("				r[k] = iVal")
		pn("			}")
		pn("		}")
		pn("	}")
		pn("")
		pn("	return json.Marshal(rawList)")
		pn("}")
		pn("")
	}

	for _, a := range s.apis {
		s.generateParamType(a)
		s.generateToURLValuesFunc(a)
		s.generateParamSettersFunc(a)
		s.generateNewParamTypeFunc(a)
		s.generateHelperFuncs(a)
		s.generateNewAPICallFunc(a)
		s.generateResponseType(a)
	}

	clean, err := format.Source(buf.Bytes())
	if err != nil {
		buf.WriteTo(os.Stdout)
		return buf.Bytes(), err
	}
	return clean, nil
}

func (s *service) generateParamType(a *API) {
	pn := s.pn

	pn("type %s struct {", capitalize(a.Name+"Params"))
	pn("	p map[string]interface{}")
	pn("}\n")
	return
}

func (s *service) generateToURLValuesFunc(a *API) {
	pn := s.pn

	pn("func (p *%s) toURLValues() url.Values {", capitalize(a.Name+"Params"))
	pn("	u := url.Values{}")
	pn("	if p.p == nil {")
	pn("		return u")
	pn("	}")
	for _, ap := range a.Params {
		pn("	if v, found := p.p[\"%s\"]; found {", ap.Name)
		s.generateConvertCode(s.parseParamName(ap.Name), mapType(ap.Type))
		pn("	}")
	}
	pn("	return u")
	pn("}")
	pn("")
	return
}

func (s *service) generateConvertCode(name, typ string) {
	pn := s.pn
	n := s.unparseParamName(name)

	switch typ {
	case "string":
		pn("u.Set(\"%s\", v.(string))", n)
	case "int":
		pn("vv := strconv.Itoa(v.(int))")
		pn("u.Set(\"%s\", vv)", n)
	case "int64":
		pn("vv := strconv.FormatInt(v.(int64), 10)")
		pn("u.Set(\"%s\", vv)", n)
	case "bool":
		pn("vv := strconv.FormatBool(v.(bool))")
		pn("u.Set(\"%s\", vv)", n)
	case "[]string":
		pn("vv := strings.Join(v.([]string), \",\")")
		pn("u.Set(\"%s\", vv)", n)
	case "map[string]string":
		pn("i := 0")
		pn("for k, vv := range v.(map[string]string) {")
		switch name {
		case "serviceproviderlist":
			pn("	u.Set(fmt.Sprintf(\"%s[%%d].service\", i), k)", n)
			pn("	u.Set(fmt.Sprintf(\"%s[%%d].provider\", i), vv)", n)
		default:
			pn("	u.Set(fmt.Sprintf(\"%s[%%d].key\", i), k)", n)
			pn("	u.Set(fmt.Sprintf(\"%s[%%d].value\", i), vv)", n)
		}
		pn("	i++")
		pn("}")
	}
	return
}

func (s *service) parseParamName(name string) string {
	if name != "type" {
		return name
	}
	return uncapitalize(strings.TrimSuffix(s.name, "Service")) + capitalize(name)
}

func (s *service) unparseParamName(name string) string {
	if name == uncapitalize(strings.TrimSuffix(s.name, "Service"))+"Type" {
		return "type"
	}
	return name
}

func (s *service) generateParamSettersFunc(a *API) {
	pn := s.pn
	found := make(map[string]bool)

	for _, ap := range a.Params {
		if !found[ap.Name] {
			pn("func (p *%s) Set%s(v %s) {", capitalize(a.Name+"Params"), capitalize(ap.Name), mapType(ap.Type))
			pn("	if p.p == nil {")
			pn("		p.p = make(map[string]interface{})")
			pn("	}")
			pn("	p.p[\"%s\"] = v", s.parseParamName(ap.Name))
			pn("	return")
			pn("}")
			pn("")
			found[ap.Name] = true
		}
	}
	return
}

func (s *service) generateNewParamTypeFunc(a *API) {
	p, pn := s.p, s.pn
	tn := capitalize(a.Name + "Params")
	rp := APIParams{}

	// Generate the function signature
	pn("// You should always use this function to get a new %s instance,", tn)
	pn("// as then you are sure you have configured all required params")
	p("func (s *%s) New%s(", s.name, tn)
	for _, ap := range a.Params {
		if ap.Required {
			rp = append(rp, ap)
			p("%s %s, ", s.parseParamName(ap.Name), mapType(ap.Type))
		}
	}
	pn(") *%s {", tn)

	// Generate the function body
	pn("	p := &%s{}", tn)
	pn("	p.p = make(map[string]interface{})")
	sort.Sort(rp)
	for _, ap := range rp {
		pn("	p.p[\"%s\"] = %s", s.parseParamName(ap.Name), s.parseParamName(ap.Name))
	}
	pn("	return p")
	pn("}")
	pn("")
	return
}

func (s *service) generateHelperFuncs(a *API) {
	p, pn := s.p, s.pn

	if strings.HasPrefix(a.Name, "list") {
		v, found := hasNameOrKeywordParamField(a.Params)
		if found && hasIDAndNameResponseField(a.Response) {
			ln := strings.TrimPrefix(a.Name, "list")

			// Check if ID is a required parameters and bail if so
			for _, ap := range a.Params {
				if ap.Required && s.parseParamName(ap.Name) == "id" {
					return
				}
			}

			// Generate the function signature
			pn("// This is a courtesy helper function, which in some cases may not work as expected!")
			p("func (s *%s) Get%sID(%s string, ", s.name, parseSingular(ln), v)
			for _, ap := range a.Params {
				if ap.Required {
					p("%s %s, ", s.parseParamName(ap.Name), mapType(ap.Type))
				}
			}
			if parseSingular(ln) == "Iso" {
				p("isofilter string, ")
			}
			if parseSingular(ln) == "Template" || parseSingular(ln) == "Iso" {
				p("zoneid string, ")
			}
			pn("opts ...OptionFunc) (string, int, error) {")

			// Generate the function body
			pn("	p := &List%sParams{}", ln)
			pn("	p.p = make(map[string]interface{})")
			pn("")
			pn("	p.p[\"%s\"] = %s", v, v)
			for _, ap := range a.Params {
				if ap.Required {
					pn("	p.p[\"%s\"] = %s", s.parseParamName(ap.Name), s.parseParamName(ap.Name))
				}
			}
			if parseSingular(ln) == "Iso" {
				pn("	p.p[\"isofilter\"] = isofilter")
			}
			if parseSingular(ln) == "Template" || parseSingular(ln) == "Iso" {
				pn("	p.p[\"zoneid\"] = zoneid")
			}
			pn("")
			pn("	for _, fn := range opts {")
			pn("		if err := fn(s.cs, p); err != nil {")
			pn("			return \"\", -1, err")
			pn("		}")
			pn("	}")
			pn("")
			pn("	l, err := s.List%s(p)", ln)
			pn("	if err != nil {")
			pn("		return \"\", -1, err")
			pn("	}")
			pn("")
			pn("	if l.Count == 0 {")
			pn("	  return \"\", l.Count, fmt.Errorf(\"No match found for %%s: %%+v\", %s, l)", v)
			pn("	}")
			pn("")
			pn("	if l.Count == 1 {")
			pn("	  return l.%s[0].Id, l.Count, nil", ln)
			pn("	}")
			pn("")
			pn(" 	if l.Count > 1 {")
			pn("    for _, v := range l.%s {", ln)
			pn("      if v.Name == %s {", v)
			pn("        return v.Id, l.Count, nil")
			pn("      }")
			pn("    }")
			pn("	}")
			pn("  return \"\", l.Count, fmt.Errorf(\"Could not find an exact match for %%s: %%+v\", %s, l)", v)
			pn("}\n")
			pn("")

			if hasIDParamField(a.Params) {
				// Generate the function signature
				pn("// This is a courtesy helper function, which in some cases may not work as expected!")
				p("func (s *%s) Get%sByName(name string, ", s.name, parseSingular(ln))
				for _, ap := range a.Params {
					if ap.Required {
						p("%s %s, ", s.parseParamName(ap.Name), mapType(ap.Type))
					}
				}
				if parseSingular(ln) == "Iso" {
					p("isofilter string, ")
				}
				if parseSingular(ln) == "Template" || parseSingular(ln) == "Iso" {
					p("zoneid string, ")
				}
				pn("opts ...OptionFunc) (*%s, int, error) {", parseSingular(ln))

				// Generate the function body
				p("  id, count, err := s.Get%sID(name, ", parseSingular(ln))
				for _, ap := range a.Params {
					if ap.Required {
						p("%s, ", s.parseParamName(ap.Name))
					}
				}
				if parseSingular(ln) == "Iso" {
					p("isofilter, ")
				}
				if parseSingular(ln) == "Template" || parseSingular(ln) == "Iso" {
					p("zoneid, ")
				}
				pn("opts...)")
				pn("  if err != nil {")
				pn("    return nil, count, err")
				pn("  }")
				pn("")
				p("  r, count, err := s.Get%sByID(id, ", parseSingular(ln))
				for _, ap := range a.Params {
					if ap.Required {
						p("%s, ", s.parseParamName(ap.Name))
					}
				}
				pn("opts...)")
				pn("  if err != nil {")
				pn("    return nil, count, err")
				pn("  }")
				pn("	return r, count, nil")
				pn("}")
				pn("")
			}
		}

		if hasIDParamField(a.Params) {
			ln := strings.TrimPrefix(a.Name, "list")

			// Generate the function signature
			pn("// This is a courtesy helper function, which in some cases may not work as expected!")
			p("func (s *%s) Get%sByID(id string, ", s.name, parseSingular(ln))
			for _, ap := range a.Params {
				if ap.Required && s.parseParamName(ap.Name) != "id" {
					p("%s %s, ", s.parseParamName(ap.Name), mapType(ap.Type))
				}
			}
			if ln == "LoadBalancerRuleInstances" {
				pn("opts ...OptionFunc) (*VirtualMachine, int, error) {")
			} else {
				pn("opts ...OptionFunc) (*%s, int, error) {", parseSingular(ln))
			}

			// Generate the function body
			pn("	p := &List%sParams{}", ln)
			pn("	p.p = make(map[string]interface{})")
			pn("")
			pn("	p.p[\"id\"] = id")
			for _, ap := range a.Params {
				if ap.Required && s.parseParamName(ap.Name) != "id" {
					pn("	p.p[\"%s\"] = %s", s.parseParamName(ap.Name), s.parseParamName(ap.Name))
				}
			}
			pn("")
			pn("	for _, fn := range opts {")
			pn("		if err := fn(s.cs, p); err != nil {")
			pn("			return nil, -1, err")
			pn("		}")
			pn("	}")
			pn("")
			pn("	l, err := s.List%s(p)", ln)
			pn("	if err != nil {")
			pn("		if strings.Contains(err.Error(), fmt.Sprintf(")
			pn("			\"Invalid parameter id value=%%s due to incorrect long value format, \"+")
			pn("				\"or entity does not exist\", id)) {")
			pn("			return nil, 0, fmt.Errorf(\"No match found for %%s: %%+v\", id, l)")
			pn("		}")
			pn("		return nil, -1, err")
			pn("	}")
			pn("")
			pn("	if l.Count == 0 {")
			pn("	  return nil, l.Count, fmt.Errorf(\"No match found for %%s: %%+v\", id, l)")
			pn("	}")
			pn("")
			pn("	if l.Count == 1 {")
			pn("	  return l.%s[0], l.Count, nil", ln)
			pn("	}")
			pn("  return nil, l.Count, fmt.Errorf(\"There is more then one result for %s UUID: %%s!\", id)", parseSingular(ln))
			pn("}\n")
			pn("")
		}
	}
	return
}

func hasNameOrKeywordParamField(params APIParams) (v string, found bool) {
	for _, p := range params {
		if p.Name == "keyword" && mapType(p.Type) == "string" {
			v = "keyword"
			found = true
		}
		if p.Name == "name" && mapType(p.Type) == "string" {
			return "name", true
		}

	}
	return v, found
}

func hasIDParamField(params APIParams) bool {
	for _, p := range params {
		if p.Name == "id" && mapType(p.Type) == "string" {
			return true
		}
	}
	return false
}

func hasIDAndNameResponseField(resp APIResponses) bool {
	id := false
	name := false

	for _, r := range resp {
		if r.Name == "id" && mapType(r.Type) == "string" {
			id = true
		}
		if r.Name == "name" && mapType(r.Type) == "string" {
			name = true
		}
	}
	return id && name
}

func (s *service) generateNewAPICallFunc(a *API) {
	pn := s.pn
	n := capitalize(a.Name)

	// Generate the function signature
	pn("// %s", a.Description)
	pn("func (s *%s) %s(p *%s) (*%s, error) {", s.name, n, n+"Params", n+"Response")

	// Generate the function body
	if n == "QueryAsyncJobResult" {
		pn("	var resp json.RawMessage")
		pn("	var err error")
		pn("")
		pn("	// We should be able to retry on failure as this call is idempotent")
		pn("	for i := 0; i < 3; i++ {")
		pn("		resp, err = s.cs.newRequest(\"%s\", p.toURLValues())", a.Name)
		pn("		if err == nil {")
		pn("			break")
		pn("		}")
		pn("		time.Sleep(500 * time.Millisecond)")
		pn("	}")
	} else {
		pn("	resp, err := s.cs.newRequest(\"%s\", p.toURLValues())", a.Name)
	}
	pn("	if err != nil {")
	pn("		return nil, err")
	pn("	}")
	pn("")
	switch n {
	case "CreateNetwork", "CreateNetworkOffering", "CreateServiceOffering", "CreateSSHKeyPair", "RegisterSSHKeyPair":
		pn("	if resp, err = getRawValue(resp); err != nil {")
		pn("		return nil, err")
		pn("	}")
		pn("")
	}
	if s.name == "FirewallService" {
		pn("	resp, err = convertFirewallServiceResponse(resp)")
		pn("	if err != nil {")
		pn("		return nil, err")
		pn("	}")
		pn("")
	}
	pn("	var r %s", n+"Response")
	pn("	if err := json.Unmarshal(resp, &r); err != nil {")
	pn("		return nil, err")
	pn("	}")
	if a.Isasync {
		pn("")
		pn("	// If we have a async client, we need to wait for the async result")
		pn("	if s.cs.async {")
		pn("		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)")
		pn("		if err != nil {")
		pn("			if err == AsyncTimeoutErr {")
		pn("				return &r, err")
		pn("			}")
		pn("			return nil, err")
		pn("		}")
		pn("")
		if !isSuccessOnlyResponse(a.Response) {
			pn("		b, err = getRawValue(b)")
			pn("		if err != nil {")
			pn("		  return nil, err")
			pn("		}")
			pn("")
		}
		if s.name == "FirewallService" {
			pn("		b, err = convertFirewallServiceResponse(b)")
			pn("		if err != nil {")
			pn("			return nil, err")
			pn("		}")
			pn("")
		}
		pn("		if err := json.Unmarshal(b, &r); err != nil {")
		pn("			return nil, err")
		pn("		}")
		pn("	}")
	}
	pn("	return &r, nil")
	pn("}")
	pn("")
}

func isSuccessOnlyResponse(resp APIResponses) bool {
	success := false
	displaytext := false

	for _, r := range resp {
		if r.Name == "displaytext" {
			displaytext = true
		}
		if r.Name == "success" {
			success = true
		}
	}
	return displaytext && success
}

func (s *service) generateResponseType(a *API) {
	pn := s.pn
	tn := capitalize(a.Name + "Response")
	ln := capitalize(strings.TrimPrefix(a.Name, "list"))

	// If this is a 'list' response, we need an seperate list struct. There seem to be other
	// types of responses that also need a seperate list struct, so checking on exact matches
	// for those once.
	if strings.HasPrefix(a.Name, "list") || a.Name == "registerTemplate" {
		pn("type %s struct {", tn)
		pn("	Count int `json:\"count\"`")

		// This nasty check is for some specific response that do not behave consistent
		switch a.Name {
		case "listAsyncJobs":
			pn("	%s []*%s `json:\"%s\"`", ln, parseSingular(ln), "asyncjobs")
		case "listEgressFirewallRules":
			pn("	%s []*%s `json:\"%s\"`", ln, parseSingular(ln), "firewallrule")
		case "listLoadBalancerRuleInstances":
			pn("	LBRuleVMIDIPs []*%s `json:\"%s,omitempty\"`", parseSingular(ln), "lbrulevmidip")
			pn("	LoadBalancerRuleInstances []*VirtualMachine `json:\"%s,omitempty\"`", strings.ToLower(parseSingular(ln)))
		case "registerTemplate":
			pn("	%s []*%s `json:\"%s\"`", ln, parseSingular(ln), "template")
		default:
			pn("	%s []*%s `json:\"%s\"`", ln, parseSingular(ln), strings.ToLower(parseSingular(ln)))
		}
		pn("}")
		pn("")
		tn = parseSingular(ln)
	}

	pn("type %s struct {", tn)
	if a.Isasync {
		pn("	JobID string `json:\"jobid,omitempty\"`")
	}
	sort.Sort(a.Response)
	s.recusiveGenerateResponseType(a.Response, a.Isasync)
	pn("}")
	pn("")
	return
}

func parseSingular(n string) string {
	if strings.HasSuffix(n, "ies") {
		return strings.TrimSuffix(n, "ies") + "y"
	}
	if strings.HasSuffix(n, "sses") {
		return strings.TrimSuffix(n, "es")
	}
	return strings.TrimSuffix(n, "s")
}

func (s *service) recusiveGenerateResponseType(resp APIResponses, async bool) (output string) {
	pn := s.pn
	found := make(map[string]bool)

	for _, r := range resp {
		if r.Name == "" {
			continue
		}
		if r.Name == "secondaryip" {
			pn("%s []struct {", capitalize(r.Name))
			pn("Id string `json:\"id,omitempty\"`")
			pn("Ipaddress string `json:\"ipaddress,omitempty\"`")
			pn("} `json:\"%s,omitempty\"`", r.Name)
			continue
		}
		if r.Response != nil {
			pn("%s []struct {", capitalize(r.Name))
			sort.Sort(r.Response)
			s.recusiveGenerateResponseType(r.Response, async)
			pn("} `json:\"%s,omitempty\"`", r.Name)
		} else {
			if !found[r.Name] {
				// This code is needed because the response field is different for sync and async calls :(
				if r.Name == "success" {
					if async {
						pn("%s bool `json:\"%s,omitempty\"`", capitalize(r.Name), r.Name)
					} else {
						pn("%s string `json:\"%s,omitempty\"`", capitalize(r.Name), r.Name)
					}
				} else {
					pn("%s %s `json:\"%s,omitempty\"`", capitalize(r.Name), mapType(r.Type), r.Name)
				}
				found[r.Name] = true
			}
		}
	}
	return
}

func getAllServices() (*allServices, []error, error) {
	// Get a map with all API info
	ai, err := getAPIInfo()
	if err != nil {
		return nil, nil, err
	}

	// Generate a complete set of services with their methods (APIs)
	as := &allServices{}
	errors := []error{}
	for sn, apis := range layout {
		s := &service{name: sn}
		for _, api := range apis {
			a, found := ai[api]
			if !found {
				errors = append(errors, &apiInfoNotFoundError{api})
				continue
			}
			s.apis = append(s.apis, a)
		}
		for _, apis := range s.apis {
			sort.Sort(apis.Params)
		}
		as.services = append(as.services, s)
	}
	sort.Sort(as.services)
	return as, errors, nil
}

func getAPIInfo() (map[string]*API, error) {
	var ar struct {
		ListAPIsResponse struct {
			Count int    `json:"count"`
			APIs  []*API `json:"api"`
		} `json:"listapisresponse"`
	}
	if err := json.Unmarshal([]byte(api), &ar); err != nil {
		return nil, err
	}

	// Make a map of all retrieved APIs
	ai := make(map[string]*API)
	for _, api := range ar.ListAPIsResponse.APIs {
		ai[api.Name] = api
	}
	return ai, nil
}

func sourceDir() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	outdir := path.Join(path.Dir(wd), pkg)

	if err := os.MkdirAll(outdir, 0755); err != nil {
		return "", fmt.Errorf("Failed to Mkdir %s: %v", outdir, err)
	}
	return outdir, nil
}

func mapType(t string) string {
	switch t {
	case "boolean":
		return "bool"
	case "short", "int", "integer":
		return "int"
	case "long":
		return "int64"
	case "list", "set":
		return "[]string"
	case "map":
		return "map[string]string"
	case "responseobject":
		return "json.RawMessage"
	case "uservmresponse":
		// This is a really specific abnormaly of the API
		return "*VirtualMachine"
	default:
		return "string"
	}
}

func capitalize(s string) string {
	r := []rune(s)
	r[0] = unicode.ToUpper(r[0])
	return string(r)
}

func uncapitalize(s string) string {
	r := []rune(s)
	r[0] = unicode.ToLower(r[0])
	return string(r)
}
