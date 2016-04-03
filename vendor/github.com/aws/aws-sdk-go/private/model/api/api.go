// Package api represents API abstractions for rendering service generated files.
package api

import (
	"bytes"
	"fmt"
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

	initialized bool
	imports     map[string]bool
	name        string
	path        string
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
		switch name {
		case "ElasticLoadBalancing":
			a.name = "ELB"
		case "Config":
			a.name = "ConfigService"
		default:
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
func (a *API) ShapeList() []*Shape {
	list := make([]*Shape, len(a.Shapes))
	for i, n := range a.ShapeNames() {
		list[i] = a.Shapes[n]
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
	delete(a.imports, "github.com/aws/aws-sdk-go/aws")
	a.imports["github.com/aws/aws-sdk-go/aws/awsutil"] = true
	a.imports["github.com/aws/aws-sdk-go/aws/request"] = true
	var buf bytes.Buffer
	err := tplAPI.Execute(&buf, a)
	if err != nil {
		panic(err)
	}

	code := a.importsGoCode() + strings.TrimSpace(buf.String())
	return code
}

// A tplService defines the template for the service generated code.
var tplService = template.Must(template.New("service").Parse(`
{{ .Documentation }}//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
type {{ .StructName }} struct {
	*client.Client
}

{{ if .UseInitMethods }}// Used for custom client initialization logic
var initClient func(*client.Client)

// Used for custom request initialization logic
var initRequest func(*request.Request)
{{ end }}

{{ if not .NoConstServiceNames }}
// A ServiceName is the name of the service the client will make API calls to.
const ServiceName = "{{ .Metadata.EndpointPrefix }}"
{{ end }}

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
	c := p.ClientConfig({{ if .NoConstServiceNames }}"{{ .Metadata.EndpointPrefix }}"{{ else }}ServiceName{{ end }}, cfgs...)
	return newClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *{{ .StructName }} {
    svc := &{{ .StructName }}{
    	Client: client.New(
    		cfg,
    		metadata.ClientInfo{
			ServiceName:  {{ if .NoConstServiceNames }}"{{ .Metadata.EndpointPrefix }}"{{ else }}ServiceName{{ end }}, {{ if ne .Metadata.SigningName "" }}
			SigningName: "{{ .Metadata.SigningName }}",{{ end }}
			SigningRegion: signingRegion,
			Endpoint:     endpoint,
			APIVersion:   "{{ .Metadata.APIVersion }}",
{{ if eq .Metadata.Protocol "json" }}JSONVersion:  "{{ .Metadata.JSONVersion }}",
			TargetPrefix: "{{ .Metadata.TargetPrefix }}",
{{ end }}
    		},
    		handlers,
    	),
    }

	// Handlers
	svc.Handlers.Sign.PushBack({{if eq .Metadata.SignatureVersion "v2"}}v2{{else}}v4{{end}}.Sign)
	{{if eq .Metadata.SignatureVersion "v2"}}svc.Handlers.Sign.PushBackNamed(corehandlers.BuildContentLengthHandler)
	{{end}}svc.Handlers.Build.PushBack({{ .ProtocolPackage }}.Build)
	svc.Handlers.Unmarshal.PushBack({{ .ProtocolPackage }}.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack({{ .ProtocolPackage }}.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack({{ .ProtocolPackage }}.UnmarshalError)

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
		a.imports["github.com/aws/aws-sdk-go/private/signer/v4"] = true
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
	for _, o := range a.OperationList() {
		exs = append(exs, o.Example())
	}

	code := fmt.Sprintf("import (\n%q\n%q\n%q\n\n%q\n%q\n%q\n)\n\n"+
		"var _ time.Duration\nvar _ bytes.Buffer\n\n%s",
		"bytes",
		"fmt",
		"time",
		"github.com/aws/aws-sdk-go/aws",
		"github.com/aws/aws-sdk-go/aws/session",
		"github.com/aws/aws-sdk-go/service/"+a.PackageName(),
		strings.Join(exs, "\n\n"),
	)
	return code
}

// A tplInterface defines the template for the service interface type.
var tplInterface = template.Must(template.New("interface").Parse(`
// {{ .StructName }}API is the interface type for {{ .PackageName }}.{{ .StructName }}.
type {{ .StructName }}API interface {
    {{ range $_, $o := .OperationList }}
        {{ $o.InterfaceSignature }}
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
		"github.com/aws/aws-sdk-go/aws/request":                true,
		"github.com/aws/aws-sdk-go/service/" + a.PackageName(): true,
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
