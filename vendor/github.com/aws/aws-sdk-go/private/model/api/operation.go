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
	API           *API `json:"-"`
	ExportedName  string
	Name          string
	Documentation string
	HTTP          HTTPInfo
	InputRef      ShapeRef   `json:"input"`
	OutputRef     ShapeRef   `json:"output"`
	ErrorRefs     []ShapeRef `json:"errors"`
	Paginator     *Paginator
	Deprecated    bool   `json:"deprecated"`
	AuthType      string `json:"authtype"`
	imports       map[string]bool
}

// A HTTPInfo defines the method of HTTP request for the Operation.
type HTTPInfo struct {
	Method       string
	RequestURI   string
	ResponseCode uint
}

// HasInput returns if the Operation accepts an input paramater
func (o *Operation) HasInput() bool {
	return o.InputRef.ShapeName != ""
}

// HasOutput returns if the Operation accepts an output parameter
func (o *Operation) HasOutput() bool {
	return o.OutputRef.ShapeName != ""
}

func (o *Operation) GetSigner() string {
	if o.AuthType == "v4-unsigned-body" {
		o.API.imports["github.com/aws/aws-sdk-go/aws/signer/v4"] = true
	}

	buf := bytes.NewBuffer(nil)

	switch o.AuthType {
	case "none":
		buf.WriteString("req.Config.Credentials = credentials.AnonymousCredentials")
	case "v4-unsigned-body":
		buf.WriteString("req.Handlers.Sign.Remove(v4.SignRequestHandler)\n")
		buf.WriteString("handler := v4.BuildNamedHandler(\"v4.CustomSignerHandler\", v4.WithUnsignedPayload)\n")
		buf.WriteString("req.Handlers.Sign.PushFrontNamed(handler)")
	}

	buf.WriteString("\n")
	return buf.String()
}

// tplOperation defines a template for rendering an API Operation
var tplOperation = template.Must(template.New("operation").Funcs(template.FuncMap{
	"GetCrosslinkURL": GetCrosslinkURL,
}).Parse(`
const op{{ .ExportedName }} = "{{ .Name }}"

// {{ .ExportedName }}Request generates a "aws/request.Request" representing the
// client's request for the {{ .ExportedName }} operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
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
{{ $crosslinkURL := GetCrosslinkURL $.API.BaseCrosslinkURL $.API.Metadata.UID $.ExportedName -}}
{{ if ne $crosslinkURL "" -}} 
//
// Please also see {{ $crosslinkURL }}
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
	req = c.newRequest(op, input, output){{ if eq .OutputRef.Shape.Placeholder true }}
	req.Handlers.Unmarshal.Remove({{ .API.ProtocolPackage }}.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler){{ end }}
	{{ if ne .AuthType "" }}{{ .GetSigner }}{{ end -}}
	return
}

// {{ .ExportedName }} API operation for {{ .API.Metadata.ServiceFullName }}.
{{ if .Documentation -}}
//
{{ .Documentation }}
{{ end -}}
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for {{ .API.Metadata.ServiceFullName }}'s
// API operation {{ .ExportedName }} for usage and error information.
{{ if .ErrorRefs -}}
//
// Returned Error Codes:
{{ range $_, $err := .ErrorRefs -}}
//   * {{ $err.Shape.ErrorCodeName }} "{{ $err.Shape.ErrorName}}"
{{ if $err.Docstring -}}
{{ $err.IndentedDocstring }}
{{ end -}}
//
{{ end -}}
{{ end -}}
{{ $crosslinkURL := GetCrosslinkURL $.API.BaseCrosslinkURL $.API.Metadata.UID $.ExportedName -}}
{{ if ne $crosslinkURL "" -}} 
// Please also see {{ $crosslinkURL }}
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
//        func(page {{ .OutputRef.GoType }}, lastPage bool) bool {
//            pageNum++
//            fmt.Println(page)
//            return pageNum <= 3
//        })
//
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
func (c *{{ .API.StructName }}) {{ .ExportedName }}PagesWithContext(` +
	`ctx aws.Context, ` +
	`input {{ .InputRef.GoType }}, ` +
	`fn func({{ .OutputRef.GoType }}, bool) bool, ` +
	`opts ...request.Option) error {
	p := request.Pagination {
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

	cont := true
	for p.Next() && cont {
		cont = fn(p.Page().({{ .OutputRef.GoType }}), !p.HasNextPage())
	}
	return p.Err()
}
{{ end }}
`))

// GoCode returns a string of rendered GoCode for this Operation
func (o *Operation) GoCode() string {
	var buf bytes.Buffer
	err := tplOperation.Execute(&buf, o)
	if err != nil {
		panic(err)
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
			o.imports["github.com/aws/aws-sdk-go/service/"+strings.Split(o.InputRef.GoTypeElem(), ".")[0]] = true
			return fmt.Sprintf("var params *%s", o.InputRef.GoTypeElem())
		}
		return fmt.Sprintf("var params *%s.%s",
			o.API.PackageName(), o.InputRef.GoTypeElem())
	}
	e := example{o, map[string]int{}}
	return "params := " + e.traverseAny(o.InputRef.Shape, false, false)
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
