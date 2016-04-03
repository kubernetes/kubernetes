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
	InputRef      ShapeRef `json:"input"`
	OutputRef     ShapeRef `json:"output"`
	Paginator     *Paginator
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

// tplOperation defines a template for rendering an API Operation
var tplOperation = template.Must(template.New("operation").Parse(`
const op{{ .ExportedName }} = "{{ .Name }}"

// {{ .ExportedName }}Request generates a request for the {{ .ExportedName }} operation.
func (c *{{ .API.StructName }}) {{ .ExportedName }}Request(` +
	`input {{ .InputRef.GoType }}) (req *request.Request, output {{ .OutputRef.GoType }}) {
	op := &request.Operation{
		Name:       op{{ .ExportedName }},
		{{ if ne .HTTP.Method "" }}HTTPMethod: "{{ .HTTP.Method }}",
		{{ end }}{{ if ne .HTTP.RequestURI "" }}HTTPPath:   "{{ .HTTP.RequestURI }}",
		{{ end }}{{ if .Paginator }}Paginator: &request.Paginator{
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

	req = c.newRequest(op, input, output)
	output = &{{ .OutputRef.GoTypeElem }}{}
	req.Data = output
	return
}

{{ .Documentation }}func (c *{{ .API.StructName }}) {{ .ExportedName }}(` +
	`input {{ .InputRef.GoType }}) ({{ .OutputRef.GoType }}, error) {
	req, out := c.{{ .ExportedName }}Request(input)
	err := req.Send()
	return out, err
}

{{ if .Paginator }}
func (c *{{ .API.StructName }}) {{ .ExportedName }}Pages(` +
	`input {{ .InputRef.GoType }}, fn func(p {{ .OutputRef.GoType }}, lastPage bool) (shouldContinue bool)) error {
	page, _ := c.{{ .ExportedName }}Request(input)
	page.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("Paginator"))
	return page.EachPage(func(p interface{}, lastPage bool) bool {
		return fn(p.({{ .OutputRef.GoType }}), lastPage)
	})
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
{{ .ExportedName }}Request({{ .InputRef.GoTypeWithPkgName }}) (*request.Request, {{ .OutputRef.GoTypeWithPkgName }})

{{ .ExportedName }}({{ .InputRef.GoTypeWithPkgName }}) ({{ .OutputRef.GoTypeWithPkgName }}, error)
{{ if .Paginator }}
{{ .ExportedName }}Pages({{ .InputRef.GoTypeWithPkgName }}, func({{ .OutputRef.GoTypeWithPkgName }}, bool) bool) error{{ end }}
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
	svc := {{ .API.PackageName }}.New(session.New())

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
	buf.WriteString("&" + s.API.PackageName() + "." + s.GoTypeElem() + "{")
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
	t := reType.ReplaceAllString(s.GoTypeElem(), s.API.PackageName()+".$1")
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
	t := reType.ReplaceAllString(s.GoTypeElem(), s.API.PackageName()+".$1")
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
