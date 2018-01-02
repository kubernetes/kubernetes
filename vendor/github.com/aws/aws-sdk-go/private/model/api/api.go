// +build codegen

// Package api represents API abstractions for rendering service generated files.
package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/template"
)

// An API defines a service API's definition. and logic to serialize the definition.
type API struct {
	Metadata      Metadata
	Operations    map[string]*Operation
	Shapes        map[string]*Shape
	Waiters       []Waiter
	Documentation string
	Examples      Examples

	// Set to true to avoid removing unused shapes
	NoRemoveUnusedShapes bool

	// Set to true to avoid renaming to 'Input/Output' postfixed shapes
	NoRenameToplevelShapes bool

	// Set to true to ignore service/request init methods (for testing)
	NoInitMethods bool

	// Set to true to ignore String() and GoString methods (for generated tests)
	NoStringerMethods bool

	// Set to true to not generate API service name constants
	NoConstServiceNames bool

	// Set to true to not generate validation shapes
	NoValidataShapeMethods bool

	// Set to true to not generate struct field accessors
	NoGenStructFieldAccessors bool

	SvcClientImportPath string

	initialized bool
	imports     map[string]bool
	name        string
	path        string

	BaseCrosslinkURL string
}

// A Metadata is the metadata about an API's definition.
type Metadata struct {
	APIVersion          string
	EndpointPrefix      string
	SigningName         string
	ServiceAbbreviation string
	ServiceFullName     string
	SignatureVersion    string
	JSONVersion         string
	TargetPrefix        string
	Protocol            string
	UID                 string
	EndpointsID         string

	NoResolveEndpoint bool
}

var serviceAliases map[string]string

func Bootstrap() error {
	b, err := ioutil.ReadFile(filepath.Join("..", "models", "customizations", "service-aliases.json"))
	if err != nil {
		return err
	}

	return json.Unmarshal(b, &serviceAliases)
}

// PackageName name of the API package
func (a *API) PackageName() string {
	return strings.ToLower(a.StructName())
}

// InterfacePackageName returns the package name for the interface.
func (a *API) InterfacePackageName() string {
	return a.PackageName() + "iface"
}

var nameRegex = regexp.MustCompile(`^Amazon|AWS\s*|\(.*|\s+|\W+`)

// StructName returns the struct name for a given API.
func (a *API) StructName() string {
	if a.name == "" {
		name := a.Metadata.ServiceAbbreviation
		if name == "" {
			name = a.Metadata.ServiceFullName
		}

		name = nameRegex.ReplaceAllString(name, "")

		a.name = name
		if name, ok := serviceAliases[strings.ToLower(name)]; ok {
			a.name = name
		}
	}
	return a.name
}

// UseInitMethods returns if the service's init method should be rendered.
func (a *API) UseInitMethods() bool {
	return !a.NoInitMethods
}

// NiceName returns the human friendly API name.
func (a *API) NiceName() string {
	if a.Metadata.ServiceAbbreviation != "" {
		return a.Metadata.ServiceAbbreviation
	}
	return a.Metadata.ServiceFullName
}

// ProtocolPackage returns the package name of the protocol this API uses.
func (a *API) ProtocolPackage() string {
	switch a.Metadata.Protocol {
	case "json":
		return "jsonrpc"
	case "ec2":
		return "ec2query"
	default:
		return strings.Replace(a.Metadata.Protocol, "-", "", -1)
	}
}

// OperationNames returns a slice of API operations supported.
func (a *API) OperationNames() []string {
	i, names := 0, make([]string, len(a.Operations))
	for n := range a.Operations {
		names[i] = n
		i++
	}
	sort.Strings(names)
	return names
}

// OperationList returns a slice of API operation pointers
func (a *API) OperationList() []*Operation {
	list := make([]*Operation, len(a.Operations))
	for i, n := range a.OperationNames() {
		list[i] = a.Operations[n]
	}
	return list
}

// OperationHasOutputPlaceholder returns if any of the API operation input
// or output shapes are place holders.
func (a *API) OperationHasOutputPlaceholder() bool {
	for _, op := range a.Operations {
		if op.OutputRef.Shape.Placeholder {
			return true
		}
	}
	return false
}

// ShapeNames returns a slice of names for each shape used by the API.
func (a *API) ShapeNames() []string {
	i, names := 0, make([]string, len(a.Shapes))
	for n := range a.Shapes {
		names[i] = n
		i++
	}
	sort.Strings(names)
	return names
}

// ShapeList returns a slice of shape pointers used by the API.
//
// Will exclude error shapes from the list of shapes returned.
func (a *API) ShapeList() []*Shape {
	list := make([]*Shape, 0, len(a.Shapes))
	for _, n := range a.ShapeNames() {
		// Ignore error shapes in list
		if s := a.Shapes[n]; !s.IsError {
			list = append(list, s)
		}
	}
	return list
}

// ShapeListErrors returns a list of the errors defined by the API model
func (a *API) ShapeListErrors() []*Shape {
	list := []*Shape{}
	for _, n := range a.ShapeNames() {
		// Ignore error shapes in list
		if s := a.Shapes[n]; s.IsError {
			list = append(list, s)
		}
	}
	return list
}

// resetImports resets the import map to default values.
func (a *API) resetImports() {
	a.imports = map[string]bool{
		"github.com/aws/aws-sdk-go/aws": true,
	}
}

// importsGoCode returns the generated Go import code.
func (a *API) importsGoCode() string {
	if len(a.imports) == 0 {
		return ""
	}

	corePkgs, extPkgs := []string{}, []string{}
	for i := range a.imports {
		if strings.Contains(i, ".") {
			extPkgs = append(extPkgs, i)
		} else {
			corePkgs = append(corePkgs, i)
		}
	}
	sort.Strings(corePkgs)
	sort.Strings(extPkgs)

	code := "import (\n"
	for _, i := range corePkgs {
		code += fmt.Sprintf("\t%q\n", i)
	}
	if len(corePkgs) > 0 {
		code += "\n"
	}
	for _, i := range extPkgs {
		code += fmt.Sprintf("\t%q\n", i)
	}
	code += ")\n\n"
	return code
}

// A tplAPI is the top level template for the API
var tplAPI = template.Must(template.New("api").Parse(`
{{ range $_, $o := .OperationList }}
{{ $o.GoCode }}

{{ end }}

{{ range $_, $s := .ShapeList }}
{{ if and $s.IsInternal (eq $s.Type "structure") }}{{ $s.GoCode }}{{ end }}

{{ end }}

{{ range $_, $s := .ShapeList }}
{{ if $s.IsEnum }}{{ $s.GoCode }}{{ end }}

{{ end }}
`))

// APIGoCode renders the API in Go code. Returning it as a string
func (a *API) APIGoCode() string {
	a.resetImports()
	a.imports["github.com/aws/aws-sdk-go/aws/awsutil"] = true
	a.imports["github.com/aws/aws-sdk-go/aws/request"] = true
	if a.OperationHasOutputPlaceholder() {
		a.imports["github.com/aws/aws-sdk-go/private/protocol/"+a.ProtocolPackage()] = true
		a.imports["github.com/aws/aws-sdk-go/private/protocol"] = true
	}

	for _, op := range a.Operations {
		if op.AuthType == "none" {
			a.imports["github.com/aws/aws-sdk-go/aws/credentials"] = true
			break
		}
	}

	var buf bytes.Buffer
	err := tplAPI.Execute(&buf, a)
	if err != nil {
		panic(err)
	}

	code := a.importsGoCode() + strings.TrimSpace(buf.String())
	return code
}

var noCrossLinkServices = map[string]struct{}{
	"apigateway":        {},
	"budgets":           {},
	"cloudsearch":       {},
	"cloudsearchdomain": {},
	"elastictranscoder": {},
	"es":                {},
	"glacier":           {},
	"importexport":      {},
	"iot":               {},
	"iot-data":          {},
	"machinelearning":   {},
	"rekognition":       {},
	"sdb":               {},
	"swf":               {},
}

// GetCrosslinkURL returns the crosslinking URL for the shape based on the name and
// uid provided. Empty string is returned if no crosslink link could be determined.
func GetCrosslinkURL(baseURL, uid string, params ...string) string {
	if uid == "" || baseURL == "" {
		return ""
	}

	if _, ok := noCrossLinkServices[strings.ToLower(serviceIDFromUID(uid))]; ok {
		return ""
	}

	return strings.Join(append([]string{baseURL, "goto", "WebAPI", uid}, params...), "/")
}

func serviceIDFromUID(uid string) string {
	found := 0
	i := len(uid) - 1
	for ; i >= 0; i-- {
		if uid[i] == '-' {
			found++
		}
		// Terminate after the date component is found, e.g. es-2017-11-11
		if found == 3 {
			break
		}
	}

	return uid[0:i]
}

// APIName returns the API's service name.
func (a *API) APIName() string {
	return a.name
}

var tplServiceDoc = template.Must(template.New("service docs").Funcs(template.FuncMap{
	"GetCrosslinkURL": GetCrosslinkURL,
}).
	Parse(`
// Package {{ .PackageName }} provides the client and types for making API
// requests to {{ .Metadata.ServiceFullName }}.
{{ if .Documentation -}}
//
{{ .Documentation }}
{{ end -}}
{{ $crosslinkURL := GetCrosslinkURL $.BaseCrosslinkURL $.Metadata.UID -}}
{{ if $crosslinkURL -}}
//
// See {{ $crosslinkURL }} for more information on this service.
{{ end -}}
//
// See {{ .PackageName }} package documentation for more information.
// https://docs.aws.amazon.com/sdk-for-go/api/service/{{ .PackageName }}/
//
// Using the Client
//
// To {{ .Metadata.ServiceFullName }} with the SDK use the New function to create
// a new service client. With that client you can make API requests to the service.
// These clients are safe to use concurrently.
//
// See the SDK's documentation for more information on how to use the SDK.
// https://docs.aws.amazon.com/sdk-for-go/api/
// 
// See aws.Config documentation for more information on configuring SDK clients.
// https://docs.aws.amazon.com/sdk-for-go/api/aws/#Config
//
// See the {{ .Metadata.ServiceFullName }} client {{ .StructName }} for more
// information on creating client for this service.
// https://docs.aws.amazon.com/sdk-for-go/api/service/{{ .PackageName }}/#New
`))

// A tplService defines the template for the service generated code.
var tplService = template.Must(template.New("service").Funcs(template.FuncMap{
	"ServiceNameValue": func(a *API) string {
		if a.NoConstServiceNames {
			return fmt.Sprintf("%q", a.Metadata.EndpointPrefix)
		}
		return "ServiceName"
	},
	"EndpointsIDConstValue": func(a *API) string {
		if a.NoConstServiceNames {
			return fmt.Sprintf("%q", a.Metadata.EndpointPrefix)
		}
		if a.Metadata.EndpointPrefix == a.Metadata.EndpointsID {
			return "ServiceName"
		}
		return fmt.Sprintf("%q", a.Metadata.EndpointsID)
	},
	"EndpointsIDValue": func(a *API) string {
		if a.NoConstServiceNames {
			return fmt.Sprintf("%q", a.Metadata.EndpointPrefix)
		}

		return "EndpointsID"
	},
}).Parse(`
// {{ .StructName }} provides the API operation methods for making requests to
// {{ .Metadata.ServiceFullName }}. See this package's package overview docs
// for details on the service.
//
// {{ .StructName }} methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type {{ .StructName }} struct {
	*client.Client
}

{{ if .UseInitMethods }}// Used for custom client initialization logic
var initClient func(*client.Client)

// Used for custom request initialization logic
var initRequest func(*request.Request)
{{ end }}


{{ if not .NoConstServiceNames -}}
// Service information constants
const (
	ServiceName = "{{ .Metadata.EndpointPrefix }}" // Service endpoint prefix API calls made to.
	EndpointsID = {{ EndpointsIDConstValue . }} // Service ID for Regions and Endpoints metadata.
)
{{- end }}

// New creates a new instance of the {{ .StructName }} client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a {{ .StructName }} client from just a session.
//     svc := {{ .PackageName }}.New(mySession)
//
//     // Create a {{ .StructName }} client with additional configuration
//     svc := {{ .PackageName }}.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func New(p client.ConfigProvider, cfgs ...*aws.Config) *{{ .StructName }} {
	{{ if .Metadata.NoResolveEndpoint -}}
		var c client.Config
		if v, ok := p.(client.ConfigNoResolveEndpointProvider); ok {
			c = v.ClientConfigNoResolveEndpoint(cfgs...)
		} else {
			c = p.ClientConfig({{ EndpointsIDValue . }}, cfgs...)
		}
	{{- else -}}
		c := p.ClientConfig({{ EndpointsIDValue . }}, cfgs...)
	{{- end }}
	return newClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *{{ .StructName }} {
	{{- if .Metadata.SigningName }}
		if len(signingName) == 0 {
			signingName = "{{ .Metadata.SigningName }}"
		}
	{{- end }}
    svc := &{{ .StructName }}{
    	Client: client.New(
    		cfg,
    		metadata.ClientInfo{
			ServiceName: {{ ServiceNameValue . }},
			SigningName: signingName,
			SigningRegion: signingRegion,
			Endpoint:     endpoint,
			APIVersion:   "{{ .Metadata.APIVersion }}",
			{{ if .Metadata.JSONVersion -}}
				JSONVersion:  "{{ .Metadata.JSONVersion }}",
			{{- end }}
			{{ if .Metadata.TargetPrefix -}}
				TargetPrefix: "{{ .Metadata.TargetPrefix }}",
			{{- end }}
    		},
    		handlers,
    	),
    }

	// Handlers
	svc.Handlers.Sign.PushBackNamed({{if eq .Metadata.SignatureVersion "v2"}}v2{{else}}v4{{end}}.SignRequestHandler)
	{{- if eq .Metadata.SignatureVersion "v2" }}
		svc.Handlers.Sign.PushBackNamed(corehandlers.BuildContentLengthHandler)
	{{- end }}
	svc.Handlers.Build.PushBackNamed({{ .ProtocolPackage }}.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed({{ .ProtocolPackage }}.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed({{ .ProtocolPackage }}.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed({{ .ProtocolPackage }}.UnmarshalErrorHandler)

	{{ if .UseInitMethods }}// Run custom client initialization if present
	if initClient != nil {
		initClient(svc.Client)
	}
	{{ end  }}

	return svc
}

// newRequest creates a new request for a {{ .StructName }} operation and runs any
// custom request initialization.
func (c *{{ .StructName }}) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	{{ if .UseInitMethods }}// Run custom request initialization if present
	if initRequest != nil {
		initRequest(req)
	}
	{{ end }}

	return req
}
`))

// ServicePackageDoc generates the contents of the doc file for the service.
//
// Will also read in the custom doc templates for the service if found.
func (a *API) ServicePackageDoc() string {
	a.imports = map[string]bool{}

	var buf bytes.Buffer
	if err := tplServiceDoc.Execute(&buf, a); err != nil {
		panic(err)
	}

	return buf.String()
}

// ServiceGoCode renders service go code. Returning it as a string.
func (a *API) ServiceGoCode() string {
	a.resetImports()
	a.imports["github.com/aws/aws-sdk-go/aws/client"] = true
	a.imports["github.com/aws/aws-sdk-go/aws/client/metadata"] = true
	a.imports["github.com/aws/aws-sdk-go/aws/request"] = true
	if a.Metadata.SignatureVersion == "v2" {
		a.imports["github.com/aws/aws-sdk-go/private/signer/v2"] = true
		a.imports["github.com/aws/aws-sdk-go/aws/corehandlers"] = true
	} else {
		a.imports["github.com/aws/aws-sdk-go/aws/signer/v4"] = true
	}
	a.imports["github.com/aws/aws-sdk-go/private/protocol/"+a.ProtocolPackage()] = true

	var buf bytes.Buffer
	err := tplService.Execute(&buf, a)
	if err != nil {
		panic(err)
	}

	code := a.importsGoCode() + buf.String()
	return code
}

// ExampleGoCode renders service example code. Returning it as a string.
func (a *API) ExampleGoCode() string {
	exs := []string{}
	imports := map[string]bool{}
	for _, o := range a.OperationList() {
		o.imports = map[string]bool{}
		exs = append(exs, o.Example())
		for k, v := range o.imports {
			imports[k] = v
		}
	}

	code := fmt.Sprintf("import (\n%q\n%q\n%q\n\n%q\n%q\n%q\n",
		"bytes",
		"fmt",
		"time",
		"github.com/aws/aws-sdk-go/aws",
		"github.com/aws/aws-sdk-go/aws/session",
		path.Join(a.SvcClientImportPath, a.PackageName()),
	)
	for k := range imports {
		code += fmt.Sprintf("%q\n", k)
	}
	code += ")\n\n"
	code += "var _ time.Duration\nvar _ bytes.Buffer\n\n"
	code += strings.Join(exs, "\n\n")
	return code
}

// A tplInterface defines the template for the service interface type.
var tplInterface = template.Must(template.New("interface").Parse(`
// {{ .StructName }}API provides an interface to enable mocking the
// {{ .PackageName }}.{{ .StructName }} service client's API operation,
// paginators, and waiters. This make unit testing your code that calls out
// to the SDK's service client's calls easier.
//
// The best way to use this interface is so the SDK's service client's calls
// can be stubbed out for unit testing your code with the SDK without needing
// to inject custom request handlers into the SDK's request pipeline.
//
//    // myFunc uses an SDK service client to make a request to
//    // {{.Metadata.ServiceFullName}}. {{ $opts := .OperationList }}{{ $opt := index $opts 0 }}
//    func myFunc(svc {{ .InterfacePackageName }}.{{ .StructName }}API) bool {
//        // Make svc.{{ $opt.ExportedName }} request
//    }
//
//    func main() {
//        sess := session.New()
//        svc := {{ .PackageName }}.New(sess)
//
//        myFunc(svc)
//    }
//
// In your _test.go file:
//
//    // Define a mock struct to be used in your unit tests of myFunc.
//    type mock{{ .StructName }}Client struct {
//        {{ .InterfacePackageName }}.{{ .StructName }}API
//    }
//    func (m *mock{{ .StructName }}Client) {{ $opt.ExportedName }}(input {{ $opt.InputRef.GoTypeWithPkgName }}) ({{ $opt.OutputRef.GoTypeWithPkgName }}, error) {
//        // mock response/functionality
//    }
//
//    func TestMyFunc(t *testing.T) {
//        // Setup Test
//        mockSvc := &mock{{ .StructName }}Client{}
//
//        myfunc(mockSvc)
//
//        // Verify myFunc's functionality
//    }
//
// It is important to note that this interface will have breaking changes
// when the service model is updated and adds new API operations, paginators,
// and waiters. Its suggested to use the pattern above for testing, or using 
// tooling to generate mocks to satisfy the interfaces.
type {{ .StructName }}API interface {
    {{ range $_, $o := .OperationList }}
        {{ $o.InterfaceSignature }}
    {{ end }}
    {{ range $_, $w := .Waiters }}
        {{ $w.InterfaceSignature }}
    {{ end }}
}

var _ {{ .StructName }}API = (*{{ .PackageName }}.{{ .StructName }})(nil)
`))

// InterfaceGoCode returns the go code for the service's API operations as an
// interface{}. Assumes that the interface is being created in a different
// package than the service API's package.
func (a *API) InterfaceGoCode() string {
	a.resetImports()
	a.imports = map[string]bool{
		"github.com/aws/aws-sdk-go/aws":                   true,
		"github.com/aws/aws-sdk-go/aws/request":           true,
		path.Join(a.SvcClientImportPath, a.PackageName()): true,
	}

	var buf bytes.Buffer
	err := tplInterface.Execute(&buf, a)

	if err != nil {
		panic(err)
	}

	code := a.importsGoCode() + strings.TrimSpace(buf.String())
	return code
}

// NewAPIGoCodeWithPkgName returns a string of instantiating the API prefixed
// with its package name. Takes a string depicting the Config.
func (a *API) NewAPIGoCodeWithPkgName(cfg string) string {
	return fmt.Sprintf("%s.New(%s)", a.PackageName(), cfg)
}

// computes the validation chain for all input shapes
func (a *API) addShapeValidations() {
	for _, o := range a.Operations {
		resolveShapeValidations(o.InputRef.Shape)
	}
}

// Updates the source shape and all nested shapes with the validations that
// could possibly be needed.
func resolveShapeValidations(s *Shape, ancestry ...*Shape) {
	for _, a := range ancestry {
		if a == s {
			return
		}
	}

	children := []string{}
	for _, name := range s.MemberNames() {
		ref := s.MemberRefs[name]

		if s.IsRequired(name) && !s.Validations.Has(ref, ShapeValidationRequired) {
			s.Validations = append(s.Validations, ShapeValidation{
				Name: name, Ref: ref, Type: ShapeValidationRequired,
			})
		}

		if ref.Shape.Min != 0 && !s.Validations.Has(ref, ShapeValidationMinVal) {
			s.Validations = append(s.Validations, ShapeValidation{
				Name: name, Ref: ref, Type: ShapeValidationMinVal,
			})
		}

		switch ref.Shape.Type {
		case "map", "list", "structure":
			children = append(children, name)
		}
	}

	ancestry = append(ancestry, s)
	for _, name := range children {
		ref := s.MemberRefs[name]
		// Since this is a grab bag we will just continue since
		// we can't validate because we don't know the valued shape.
		if ref.JSONValue {
			continue
		}

		nestedShape := ref.Shape.NestedShape()

		var v *ShapeValidation
		if len(nestedShape.Validations) > 0 {
			v = &ShapeValidation{
				Name: name, Ref: ref, Type: ShapeValidationNested,
			}
		} else {
			resolveShapeValidations(nestedShape, ancestry...)
			if len(nestedShape.Validations) > 0 {
				v = &ShapeValidation{
					Name: name, Ref: ref, Type: ShapeValidationNested,
				}
			}
		}

		if v != nil && !s.Validations.Has(v.Ref, v.Type) {
			s.Validations = append(s.Validations, *v)
		}
	}
	ancestry = ancestry[:len(ancestry)-1]
}

// A tplAPIErrors is the top level template for the API
var tplAPIErrors = template.Must(template.New("api").Parse(`
const (
{{ range $_, $s := $.ShapeListErrors }}
	// {{ $s.ErrorCodeName }} for service response error code
	// {{ printf "%q" $s.ErrorName }}.
	{{ if $s.Docstring -}}
	//
	{{ $s.Docstring }}
	{{ end -}}
	{{ $s.ErrorCodeName }} = {{ printf "%q" $s.ErrorName }}
{{ end }}
)
`))

func (a *API) APIErrorsGoCode() string {
	var buf bytes.Buffer
	err := tplAPIErrors.Execute(&buf, a)

	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(buf.String())
}
