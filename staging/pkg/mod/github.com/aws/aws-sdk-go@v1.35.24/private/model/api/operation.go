// +build codegen

package api

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"text/template"
)

// An Operation defines a specific API Operation.
type Operation struct {
	API                 *API `json:"-"`
	ExportedName        string
	Name                string
	Documentation       string
	HTTP                HTTPInfo
	Host                string     `json:"host"`
	InputRef            ShapeRef   `json:"input"`
	OutputRef           ShapeRef   `json:"output"`
	ErrorRefs           []ShapeRef `json:"errors"`
	Paginator           *Paginator
	Deprecated          bool     `json:"deprecated"`
	DeprecatedMsg       string   `json:"deprecatedMessage"`
	AuthType            AuthType `json:"authtype"`
	imports             map[string]bool
	CustomBuildHandlers []string

	EventStreamAPI *EventStreamAPI

	IsEndpointDiscoveryOp  bool               `json:"endpointoperation"`
	EndpointDiscovery      *EndpointDiscovery `json:"endpointdiscovery"`
	Endpoint               *EndpointTrait     `json:"endpoint"`
	IsHttpChecksumRequired bool               `json:"httpChecksumRequired"`
}

// EndpointTrait provides the structure of the modeled endpoint trait, and its
// properties.
type EndpointTrait struct {
	// Specifies the hostPrefix template to prepend to the operation's request
	// endpoint host.
	HostPrefix string `json:"hostPrefix"`
}

// EndpointDiscovery represents a map of key values pairs that represents
// metadata about how a given API will make a call to the discovery endpoint.
type EndpointDiscovery struct {
	// Required indicates that for a given operation that endpoint is required.
	// Any required endpoint discovery operation cannot have endpoint discovery
	// turned off.
	Required bool `json:"required"`
}

// OperationForMethod returns the API operation name that corresponds to the
// client method name provided.
func (a *API) OperationForMethod(name string) *Operation {
	for _, op := range a.Operations {
		for _, m := range op.Methods() {
			if m == name {
				return op
			}
		}
	}

	return nil
}

// A HTTPInfo defines the method of HTTP request for the Operation.
type HTTPInfo struct {
	Method       string
	RequestURI   string
	ResponseCode uint
}

// Methods Returns a list of method names that will be generated.
func (o *Operation) Methods() []string {
	methods := []string{
		o.ExportedName,
		o.ExportedName + "Request",
		o.ExportedName + "WithContext",
	}

	if o.Paginator != nil {
		methods = append(methods, []string{
			o.ExportedName + "Pages",
			o.ExportedName + "PagesWithContext",
		}...)
	}

	return methods
}

// HasInput returns if the Operation accepts an input parameter
func (o *Operation) HasInput() bool {
	return o.InputRef.ShapeName != ""
}

// HasOutput returns if the Operation accepts an output parameter
func (o *Operation) HasOutput() bool {
	return o.OutputRef.ShapeName != ""
}

// AuthType provides the enumeration of AuthType trait.
type AuthType string

// Enumeration values for AuthType trait
const (
	NoneAuthType           AuthType = "none"
	V4UnsignedBodyAuthType AuthType = "v4-unsigned-body"
)

// ShouldSignRequestBody returns if the operation request body should be signed
// or not.
func (o *Operation) ShouldSignRequestBody() bool {
	switch o.AuthType {
	case NoneAuthType, V4UnsignedBodyAuthType:
		return false
	default:
		return true
	}
}

// GetSigner returns the signer that should be used for a API request.
func (o *Operation) GetSigner() string {
	buf := bytes.NewBuffer(nil)

	switch o.AuthType {
	case NoneAuthType:
		o.API.AddSDKImport("aws/credentials")

		buf.WriteString("req.Config.Credentials = credentials.AnonymousCredentials")
	case V4UnsignedBodyAuthType:
		o.API.AddSDKImport("aws/signer/v4")

		buf.WriteString("req.Handlers.Sign.Remove(v4.SignRequestHandler)\n")
		buf.WriteString("handler := v4.BuildNamedHandler(\"v4.CustomSignerHandler\", v4.WithUnsignedPayload)\n")
		buf.WriteString("req.Handlers.Sign.PushFrontNamed(handler)")
	}

	return buf.String()
}

// HasAccountIDMemberWithARN returns true if an account id member exists for an input shape that may take in an ARN.
func (o *Operation) HasAccountIDMemberWithARN() bool {
	return o.InputRef.Shape.HasAccountIdMemberWithARN
}

// operationTmpl defines a template for rendering an API Operation
var operationTmpl = template.Must(template.New("operation").Funcs(template.FuncMap{
	"EnableStopOnSameToken": enableStopOnSameToken,
	"GetDeprecatedMsg":      getDeprecatedMessage,
}).Parse(`
const op{{ .ExportedName }} = "{{ .Name }}"

// {{ .ExportedName }}Request generates a "aws/request.Request" representing the
// client's request for the {{ .ExportedName }} operation. The "output" return
// value will be populated with the request's response once the request completes
// successfully.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See {{ .ExportedName }} for more information on using the {{ .ExportedName }}
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the {{ .ExportedName }}Request method.
//    req, resp := client.{{ .ExportedName }}Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
{{ $crosslinkURL := $.API.GetCrosslinkURL $.ExportedName -}}
{{ if ne $crosslinkURL "" -}}
//
// See also, {{ $crosslinkURL }}
{{ end -}}
{{- if .Deprecated }}//
// Deprecated: {{ GetDeprecatedMsg .DeprecatedMsg .ExportedName }}
{{ end -}}
func (c *{{ .API.StructName }}) {{ .ExportedName }}Request(` +
	`input {{ .InputRef.GoType }}) (req *request.Request, output {{ .OutputRef.GoType }}) {
	{{ if (or .Deprecated (or .InputRef.Deprecated .OutputRef.Deprecated)) }}if c.Client.Config.Logger != nil {
		c.Client.Config.Logger.Log("This operation, {{ .ExportedName }}, has been deprecated")
	}
	op := &request.Operation{ {{ else }} op := &request.Operation{ {{ end }}
		Name:       op{{ .ExportedName }},
		{{ if ne .HTTP.Method "" }}HTTPMethod: "{{ .HTTP.Method }}",
		{{ end }}HTTPPath: {{ if ne .HTTP.RequestURI "" }}"{{ .HTTP.RequestURI }}"{{ else }}"/"{{ end }},
		{{ if .Paginator }}Paginator: &request.Paginator{
				InputTokens: {{ .Paginator.InputTokensString }},
				OutputTokens: {{ .Paginator.OutputTokensString }},
				LimitToken: "{{ .Paginator.LimitKey }}",
				TruncationToken: "{{ .Paginator.MoreResults }}",
		},
		{{ end }}
	}

	if input == nil {
		input = &{{ .InputRef.GoTypeElem }}{}
	}

	output = &{{ .OutputRef.GoTypeElem }}{}
	req = c.newRequest(op, input, output)
	{{- if ne .AuthType "" }}
		{{ .GetSigner }}
	{{- end }}

	{{- if .HasAccountIDMemberWithARN }}
		// update account id or check if provided input for account id member matches 
		// the account id present in ARN
		req.Handlers.Validate.PushFrontNamed(updateAccountIDWithARNHandler)
	{{- end }}

	{{- if .ShouldDiscardResponse -}}
		{{- $_ := .API.AddSDKImport "private/protocol" }}
		{{- $_ := .API.AddSDKImport "private/protocol" .API.ProtocolPackage }}
		req.Handlers.Unmarshal.Swap({{ .API.ProtocolPackage }}.UnmarshalHandler.Name, protocol.UnmarshalDiscardBodyHandler)
	{{- else }}
		{{- if $.EventStreamAPI }}
			{{- $esapi := $.EventStreamAPI }}

			{{- if $esapi.RequireHTTP2 }}
				req.Handlers.UnmarshalMeta.PushBack(
					protocol.RequireHTTPMinProtocol{Major:2}.Handler,
				)
			{{- end }}

			es := New{{ $esapi.Name }}()
			{{- if $esapi.Legacy }}
				req.Handlers.Unmarshal.PushBack(es.setStreamCloser)
			{{- end }}
			output.{{ $esapi.OutputMemberName }} = es

			{{- $inputStream := $esapi.InputStream }}
			{{- $outputStream := $esapi.OutputStream }}

			{{- $_ := .API.AddSDKImport "private/protocol" .API.ProtocolPackage }}
			{{- $_ := .API.AddSDKImport "private/protocol/rest" }}

			{{- if $inputStream }}

				req.Handlers.Sign.PushFront(es.setupInputPipe)
				req.Handlers.Build.PushBack(request.WithSetRequestHeaders(map[string]string{
					"Content-Type": "application/vnd.amazon.eventstream",
					"X-Amz-Content-Sha256": "STREAMING-AWS4-HMAC-SHA256-EVENTS",
				}))
				req.Handlers.Build.Swap({{ .API.ProtocolPackage }}.BuildHandler.Name, rest.BuildHandler)
				req.Handlers.Send.Swap(client.LogHTTPRequestHandler.Name, client.LogHTTPRequestHeaderHandler)
				req.Handlers.Unmarshal.PushBack(es.runInputStream)

				{{- if eq .API.Metadata.Protocol "json" }}
					es.input = input
					req.Handlers.Unmarshal.PushBack(es.sendInitialEvent)
				{{- end }}
			{{- end }}

			{{- if $outputStream }}

				req.Handlers.Send.Swap(client.LogHTTPResponseHandler.Name, client.LogHTTPResponseHeaderHandler)
				req.Handlers.Unmarshal.Swap({{ .API.ProtocolPackage }}.UnmarshalHandler.Name, rest.UnmarshalHandler)
				req.Handlers.Unmarshal.PushBack(es.runOutputStream)

				{{- if eq .API.Metadata.Protocol "json" }}
					es.output = output
					req.Handlers.Unmarshal.PushBack(es.recvInitialEvent)
				{{- end }}
			{{- end }}
			req.Handlers.Unmarshal.PushBack(es.runOnStreamPartClose)

		{{- end }}
	{{- end }}

	{{- if .EndpointDiscovery }}
		// if custom endpoint for the request is set to a non empty string,
		// we skip the endpoint discovery workflow.
		if req.Config.Endpoint == nil || *req.Config.Endpoint == "" {
			{{- if not .EndpointDiscovery.Required }}
				if aws.BoolValue(req.Config.EnableEndpointDiscovery) {
			{{- end }}
			de := discoverer{{ .API.EndpointDiscoveryOp.Name }}{
				Required: {{ .EndpointDiscovery.Required }},
				EndpointCache: c.endpointCache,
				Params: map[string]*string{
					"op": aws.String(req.Operation.Name),
					{{- range $key, $ref := .InputRef.Shape.MemberRefs -}}
						{{- if $ref.EndpointDiscoveryID -}}
							{{- if ne (len $ref.LocationName) 0 -}}
								"{{ $ref.LocationName }}": input.{{ $key }},
							{{- else }}
								"{{ $key }}": input.{{ $key }},
							{{- end }}
						{{- end }}
					{{- end }}
				},
				Client: c,
			}

			for k, v := range de.Params {
				if v == nil {
					delete(de.Params, k)
				}
			}

			req.Handlers.Build.PushFrontNamed(request.NamedHandler{
				Name: "crr.endpointdiscovery",
				Fn: de.Handler,
			})
			{{- if not .EndpointDiscovery.Required }}
				}
			{{- end }}
		}
	{{- end }}

	{{- range $_, $handler := $.CustomBuildHandlers }}
		req.Handlers.Build.PushBackNamed({{ $handler }})
	{{- end }}

	{{- if .IsHttpChecksumRequired }}
		{{- $_ := .API.AddSDKImport "private/checksum" }}
		req.Handlers.Build.PushBackNamed(request.NamedHandler{
			Name: "contentMd5Handler",
			Fn: checksum.AddBodyContentMD5Handler,
		})
	{{- end }}
	return
}

// {{ .ExportedName }} API operation for {{ .API.Metadata.ServiceFullName }}.
{{- if .Documentation }}
//
{{ .Documentation }}
{{- end }}
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for {{ .API.Metadata.ServiceFullName }}'s
// API operation {{ .ExportedName }} for usage and error information.
{{- if .ErrorRefs }}
//
// Returned Error {{ if $.API.WithGeneratedTypedErrors }}Types{{ else }}Codes{{ end }}:
{{- range $_, $err := .ErrorRefs -}}
{{- if $.API.WithGeneratedTypedErrors }}
//   * {{ $err.ShapeName }}
{{- else }}
//   * {{ $err.Shape.ErrorCodeName }} "{{ $err.Shape.ErrorName}}"
{{- end }}
{{- if $err.Docstring }}
{{ $err.IndentedDocstring }}
{{- end }}
//
{{- end }}
{{- end }}
{{ $crosslinkURL := $.API.GetCrosslinkURL $.ExportedName -}}
{{ if ne $crosslinkURL "" -}}
// See also, {{ $crosslinkURL }}
{{ end -}}
{{- if .Deprecated }}//
// Deprecated: {{ GetDeprecatedMsg .DeprecatedMsg .ExportedName }}
{{ end -}}
func (c *{{ .API.StructName }}) {{ .ExportedName }}(` +
	`input {{ .InputRef.GoType }}) ({{ .OutputRef.GoType }}, error) {
	req, out := c.{{ .ExportedName }}Request(input)
	return out, req.Send()
}

// {{ .ExportedName }}WithContext is the same as {{ .ExportedName }} with the addition of
// the ability to pass a context and additional request options.
//
// See {{ .ExportedName }} for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
{{ if .Deprecated }}//
// Deprecated: {{ GetDeprecatedMsg .DeprecatedMsg (printf "%s%s" .ExportedName "WithContext") }}
{{ end -}}
func (c *{{ .API.StructName }}) {{ .ExportedName }}WithContext(` +
	`ctx aws.Context, input {{ .InputRef.GoType }}, opts ...request.Option) ` +
	`({{ .OutputRef.GoType }}, error) {
	req, out := c.{{ .ExportedName }}Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

{{ if .Paginator }}
// {{ .ExportedName }}Pages iterates over the pages of a {{ .ExportedName }} operation,
// calling the "fn" function with the response data for each page. To stop
// iterating, return false from the fn function.
//
// See {{ .ExportedName }} method for more information on how to use this operation.
//
// Note: This operation can generate multiple requests to a service.
//
//    // Example iterating over at most 3 pages of a {{ .ExportedName }} operation.
//    pageNum := 0
//    err := client.{{ .ExportedName }}Pages(params,
//        func(page {{ .OutputRef.Shape.GoTypeWithPkgName }}, lastPage bool) bool {
//            pageNum++
//            fmt.Println(page)
//            return pageNum <= 3
//        })
//
{{ if .Deprecated }}//
// Deprecated: {{ GetDeprecatedMsg .DeprecatedMsg (printf "%s%s" .ExportedName "Pages") }}
{{ end -}}
func (c *{{ .API.StructName }}) {{ .ExportedName }}Pages(` +
	`input {{ .InputRef.GoType }}, fn func({{ .OutputRef.GoType }}, bool) bool) error {
	return c.{{ .ExportedName }}PagesWithContext(aws.BackgroundContext(), input, fn)
}

// {{ .ExportedName }}PagesWithContext same as {{ .ExportedName }}Pages except
// it takes a Context and allows setting request options on the pages.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
{{ if .Deprecated }}//
// Deprecated: {{ GetDeprecatedMsg .DeprecatedMsg (printf "%s%s" .ExportedName "PagesWithContext") }}
{{ end -}}
func (c *{{ .API.StructName }}) {{ .ExportedName }}PagesWithContext(` +
	`ctx aws.Context, ` +
	`input {{ .InputRef.GoType }}, ` +
	`fn func({{ .OutputRef.GoType }}, bool) bool, ` +
	`opts ...request.Option) error {
	p := request.Pagination {
		{{ if EnableStopOnSameToken .API.PackageName -}}EndPageOnSameToken: true,
		{{ end -}}
		NewRequest: func() (*request.Request, error) {
			var inCpy {{ .InputRef.GoType }}
			if input != nil  {
				tmp := *input
				inCpy = &tmp
			}
			req, _ := c.{{ .ExportedName }}Request(inCpy)
			req.SetContext(ctx)
			req.ApplyOptions(opts...)
			return req, nil
		},
	}

	for p.Next() {
		if !fn(p.Page().({{ .OutputRef.GoType }}), !p.HasNextPage()) {
			break
		}
	}

	return p.Err()
}
{{ end }}

{{- if .IsEndpointDiscoveryOp }}
type discoverer{{ .ExportedName }} struct {
	Client *{{ .API.StructName }}
	Required bool
	EndpointCache *crr.EndpointCache
	Params map[string]*string
	Key string
	req *request.Request
}

func (d *discoverer{{ .ExportedName }}) Discover() (crr.Endpoint, error) {
	input := &{{ .API.EndpointDiscoveryOp.InputRef.ShapeName }}{
		{{ if .API.EndpointDiscoveryOp.InputRef.Shape.HasMember "Operation" -}}
		Operation: d.Params["op"],
		{{ end -}}
		{{ if .API.EndpointDiscoveryOp.InputRef.Shape.HasMember "Identifiers" -}}
		Identifiers: d.Params,
		{{ end -}}
	}

	resp, err := d.Client.{{ .API.EndpointDiscoveryOp.Name }}(input)
	if err != nil {
		return crr.Endpoint{}, err
	}

	endpoint := crr.Endpoint{
		Key: d.Key,
	}

	for _, e := range resp.Endpoints {
		if e.Address == nil {
			continue
		}

		address := *e.Address

		var scheme string
		if idx := strings.Index(address, "://"); idx != -1 {
			scheme = address[:idx]
		}

		if len(scheme) == 0 {
			address = fmt.Sprintf("%s://%s", d.req.HTTPRequest.URL.Scheme, address)
		}

		cachedInMinutes := aws.Int64Value(e.CachePeriodInMinutes)
		u, err := url.Parse(address)
		if err != nil {
			continue
		}

		addr := crr.WeightedAddress{
			URL: u,
			Expired:  time.Now().Add(time.Duration(cachedInMinutes) * time.Minute),
		}

		endpoint.Add(addr)
	}

	d.EndpointCache.Add(endpoint)

	return endpoint, nil
}

func (d *discoverer{{ .ExportedName }}) Handler(r *request.Request) {
	endpointKey := crr.BuildEndpointKey(d.Params)
	d.Key = endpointKey
	d.req = r

	endpoint, err := d.EndpointCache.Get(d, endpointKey, d.Required)
	if err != nil {
		r.Error = err
		return
	}

	if endpoint.URL != nil && len(endpoint.URL.String()) > 0 {
		r.HTTPRequest.URL = endpoint.URL
	}
}
{{- end }}
`))

// GoCode returns a string of rendered GoCode for this Operation
func (o *Operation) GoCode() string {
	var buf bytes.Buffer

	if o.API.EndpointDiscoveryOp != nil {
		o.API.AddSDKImport("aws/crr")
		o.API.AddImport("time")
		o.API.AddImport("net/url")
		o.API.AddImport("fmt")
		o.API.AddImport("strings")
	}

	if o.Endpoint != nil && len(o.Endpoint.HostPrefix) != 0 {
		setupEndpointHostPrefix(o)
	}

	if err := operationTmpl.Execute(&buf, o); err != nil {
		panic(fmt.Sprintf("failed to render operation, %v, %v", o.ExportedName, err))
	}

	if o.EventStreamAPI != nil {
		o.API.AddSDKImport("aws/client")
		o.API.AddSDKImport("private/protocol")
		o.API.AddSDKImport("private/protocol/rest")
		o.API.AddSDKImport("private/protocol", o.API.ProtocolPackage())
		if err := renderEventStreamAPI(&buf, o); err != nil {
			panic(fmt.Sprintf("failed to render EventStreamAPI for %v, %v", o.ExportedName, err))
		}
	}

	return strings.TrimSpace(buf.String())
}

// tplInfSig defines the template for rendering an Operation's signature within an Interface definition.
var tplInfSig = template.Must(template.New("opsig").Parse(`
{{ .ExportedName }}({{ .InputRef.GoTypeWithPkgName }}) ({{ .OutputRef.GoTypeWithPkgName }}, error)
{{ .ExportedName }}WithContext(aws.Context, {{ .InputRef.GoTypeWithPkgName }}, ...request.Option) ({{ .OutputRef.GoTypeWithPkgName }}, error)
{{ .ExportedName }}Request({{ .InputRef.GoTypeWithPkgName }}) (*request.Request, {{ .OutputRef.GoTypeWithPkgName }})

{{ if .Paginator -}}
{{ .ExportedName }}Pages({{ .InputRef.GoTypeWithPkgName }}, func({{ .OutputRef.GoTypeWithPkgName }}, bool) bool) error
{{ .ExportedName }}PagesWithContext(aws.Context, {{ .InputRef.GoTypeWithPkgName }}, func({{ .OutputRef.GoTypeWithPkgName }}, bool) bool, ...request.Option) error
{{- end }}
`))

// InterfaceSignature returns a string representing the Operation's interface{}
// functional signature.
func (o *Operation) InterfaceSignature() string {
	var buf bytes.Buffer
	err := tplInfSig.Execute(&buf, o)
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(buf.String())
}

// tplExample defines the template for rendering an Operation example
var tplExample = template.Must(template.New("operationExample").Parse(`
func Example{{ .API.StructName }}_{{ .ExportedName }}() {
	sess := session.Must(session.NewSession())

	svc := {{ .API.PackageName }}.New(sess)

	{{ .ExampleInput }}
	resp, err := svc.{{ .ExportedName }}(params)

	if err != nil {
		// Print the error, cast err to awserr.Error to get the Code and
		// Message from an error.
		fmt.Println(err.Error())
		return
	}

	// Pretty-print the response data.
	fmt.Println(resp)
}
`))

// Example returns a string of the rendered Go code for the Operation
func (o *Operation) Example() string {
	var buf bytes.Buffer
	err := tplExample.Execute(&buf, o)
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(buf.String())
}

// ExampleInput return a string of the rendered Go code for an example's input parameters
func (o *Operation) ExampleInput() string {
	if len(o.InputRef.Shape.MemberRefs) == 0 {
		if strings.Contains(o.InputRef.GoTypeElem(), ".") {
			o.imports[SDKImportRoot+"service/"+strings.Split(o.InputRef.GoTypeElem(), ".")[0]] = true
			return fmt.Sprintf("var params *%s", o.InputRef.GoTypeElem())
		}
		return fmt.Sprintf("var params *%s.%s",
			o.API.PackageName(), o.InputRef.GoTypeElem())
	}
	e := example{o, map[string]int{}}
	return "params := " + e.traverseAny(o.InputRef.Shape, false, false)
}

// ShouldDiscardResponse returns if the operation should discard the response
// returned by the service.
func (o *Operation) ShouldDiscardResponse() bool {
	s := o.OutputRef.Shape
	return s.Placeholder || len(s.MemberRefs) == 0
}

// A example provides
type example struct {
	*Operation
	visited map[string]int
}

// traverseAny returns rendered Go code for the shape.
func (e *example) traverseAny(s *Shape, required, payload bool) string {
	str := ""
	e.visited[s.ShapeName]++

	switch s.Type {
	case "structure":
		str = e.traverseStruct(s, required, payload)
	case "list":
		str = e.traverseList(s, required, payload)
	case "map":
		str = e.traverseMap(s, required, payload)
	case "jsonvalue":
		str = "aws.JSONValue{\"key\": \"value\"}"
		if required {
			str += " // Required"
		}
	default:
		str = e.traverseScalar(s, required, payload)
	}

	e.visited[s.ShapeName]--

	return str
}

var reType = regexp.MustCompile(`\b([A-Z])`)

// traverseStruct returns rendered Go code for a structure type shape.
func (e *example) traverseStruct(s *Shape, required, payload bool) string {
	var buf bytes.Buffer

	if s.resolvePkg != "" {
		e.imports[s.resolvePkg] = true
		buf.WriteString("&" + s.GoTypeElem() + "{")
	} else {
		buf.WriteString("&" + s.API.PackageName() + "." + s.GoTypeElem() + "{")
	}

	if required {
		buf.WriteString(" // Required")
	}
	buf.WriteString("\n")

	req := make([]string, len(s.Required))
	copy(req, s.Required)
	sort.Strings(req)

	if e.visited[s.ShapeName] < 2 {
		for _, n := range req {
			m := s.MemberRefs[n].Shape
			p := n == s.Payload && (s.MemberRefs[n].Streaming || m.Streaming)
			buf.WriteString(n + ": " + e.traverseAny(m, true, p) + ",")
			if m.Type != "list" && m.Type != "structure" && m.Type != "map" {
				buf.WriteString(" // Required")
			}
			buf.WriteString("\n")
		}

		for _, n := range s.MemberNames() {
			if s.IsRequired(n) {
				continue
			}
			m := s.MemberRefs[n].Shape
			p := n == s.Payload && (s.MemberRefs[n].Streaming || m.Streaming)
			buf.WriteString(n + ": " + e.traverseAny(m, false, p) + ",\n")
		}
	} else {
		buf.WriteString("// Recursive values...\n")
	}

	buf.WriteString("}")
	return buf.String()
}

// traverseMap returns rendered Go code for a map type shape.
func (e *example) traverseMap(s *Shape, required, payload bool) string {
	var buf bytes.Buffer

	t := ""
	if s.resolvePkg != "" {
		e.imports[s.resolvePkg] = true
		t = s.GoTypeElem()
	} else {
		t = reType.ReplaceAllString(s.GoTypeElem(), s.API.PackageName()+".$1")
	}
	buf.WriteString(t + "{")
	if required {
		buf.WriteString(" // Required")
	}
	buf.WriteString("\n")

	if e.visited[s.ShapeName] < 2 {
		m := s.ValueRef.Shape
		buf.WriteString("\"Key\": " + e.traverseAny(m, true, false) + ",")
		if m.Type != "list" && m.Type != "structure" && m.Type != "map" {
			buf.WriteString(" // Required")
		}
		buf.WriteString("\n// More values...\n")
	} else {
		buf.WriteString("// Recursive values...\n")
	}
	buf.WriteString("}")

	return buf.String()
}

// traverseList returns rendered Go code for a list type shape.
func (e *example) traverseList(s *Shape, required, payload bool) string {
	var buf bytes.Buffer
	t := ""
	if s.resolvePkg != "" {
		e.imports[s.resolvePkg] = true
		t = s.GoTypeElem()
	} else {
		t = reType.ReplaceAllString(s.GoTypeElem(), s.API.PackageName()+".$1")
	}

	buf.WriteString(t + "{")
	if required {
		buf.WriteString(" // Required")
	}
	buf.WriteString("\n")

	if e.visited[s.ShapeName] < 2 {
		m := s.MemberRef.Shape
		buf.WriteString(e.traverseAny(m, true, false) + ",")
		if m.Type != "list" && m.Type != "structure" && m.Type != "map" {
			buf.WriteString(" // Required")
		}
		buf.WriteString("\n// More values...\n")
	} else {
		buf.WriteString("// Recursive values...\n")
	}
	buf.WriteString("}")

	return buf.String()
}

// traverseScalar returns an AWS Type string representation initialized to a value.
// Will panic if s is an unsupported shape type.
func (e *example) traverseScalar(s *Shape, required, payload bool) string {
	str := ""
	switch s.Type {
	case "integer", "long":
		str = `aws.Int64(1)`
	case "float", "double":
		str = `aws.Float64(1.0)`
	case "string", "character":
		str = `aws.String("` + s.ShapeName + `")`
	case "blob":
		if payload {
			str = `bytes.NewReader([]byte("PAYLOAD"))`
		} else {
			str = `[]byte("PAYLOAD")`
		}
	case "boolean":
		str = `aws.Bool(true)`
	case "timestamp":
		str = `aws.Time(time.Now())`
	default:
		panic("unsupported shape " + s.Type)
	}

	return str
}
