// +build codegen

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/template"
)

// WaiterAcceptor is the acceptors defined in the model the SDK will use
// to wait on resource states with.
type WaiterAcceptor struct {
	State    string
	Matcher  string
	Argument string
	Expected interface{}
}

// ExpectedString returns the string that was expected by the WaiterAcceptor
func (a *WaiterAcceptor) ExpectedString() string {
	switch a.Expected.(type) {
	case string:
		return fmt.Sprintf("%q", a.Expected)
	default:
		return fmt.Sprintf("%v", a.Expected)
	}
}

// A Waiter is an individual waiter definition.
type Waiter struct {
	Name          string
	Delay         int
	MaxAttempts   int
	OperationName string `json:"operation"`
	Operation     *Operation
	Acceptors     []WaiterAcceptor
}

// WaitersGoCode generates and returns Go code for each of the waiters of
// this API.
func (a *API) WaitersGoCode() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "import (\n%q\n\n%q\n%q\n)",
		"time",
		"github.com/aws/aws-sdk-go/aws",
		"github.com/aws/aws-sdk-go/aws/request",
	)

	for _, w := range a.Waiters {
		buf.WriteString(w.GoCode())
	}
	return buf.String()
}

// used for unmarshaling from the waiter JSON file
type waiterDefinitions struct {
	*API
	Waiters map[string]Waiter
}

// AttachWaiters reads a file of waiter definitions, and adds those to the API.
// Will panic if an error occurs.
func (a *API) AttachWaiters(filename string) {
	p := waiterDefinitions{API: a}

	f, err := os.Open(filename)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	err = json.NewDecoder(f).Decode(&p)
	if err != nil {
		panic(err)
	}

	p.setup()
}

func (p *waiterDefinitions) setup() {
	p.API.Waiters = []Waiter{}
	i, keys := 0, make([]string, len(p.Waiters))
	for k := range p.Waiters {
		keys[i] = k
		i++
	}
	sort.Strings(keys)

	for _, n := range keys {
		e := p.Waiters[n]
		n = p.ExportableName(n)
		e.Name = n
		e.OperationName = p.ExportableName(e.OperationName)
		e.Operation = p.API.Operations[e.OperationName]
		if e.Operation == nil {
			panic("unknown operation " + e.OperationName + " for waiter " + n)
		}
		p.API.Waiters = append(p.API.Waiters, e)
	}
}

var waiterTmpls = template.Must(template.New("waiterTmpls").Funcs(
	template.FuncMap{
		"titleCase": func(v string) string {
			return strings.Title(v)
		},
	},
).Parse(`
{{ define "waiter"}}
// WaitUntil{{ .Name }} uses the {{ .Operation.API.NiceName }} API operation
// {{ .OperationName }} to wait for a condition to be met before returning.
// If the condition is not met within the max attempt window, an error will
// be returned.
func (c *{{ .Operation.API.StructName }}) WaitUntil{{ .Name }}(input {{ .Operation.InputRef.GoType }}) error {
	return c.WaitUntil{{ .Name }}WithContext(aws.BackgroundContext(), input)
}

// WaitUntil{{ .Name }}WithContext is an extended version of WaitUntil{{ .Name }}.
// With the support for passing in a context and options to configure the
// Waiter and the underlying request options.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *{{ .Operation.API.StructName }}) WaitUntil{{ .Name }}WithContext(` +
	`ctx aws.Context, input {{ .Operation.InputRef.GoType }}, opts ...request.WaiterOption) error {
	w := request.Waiter{
		Name:    "WaitUntil{{ .Name }}",
		MaxAttempts: {{ .MaxAttempts }},
		Delay: request.ConstantWaiterDelay({{ .Delay }} * time.Second),
		Acceptors: []request.WaiterAcceptor{
			{{ range $_, $a := .Acceptors }}{
				State:    request.{{ titleCase .State }}WaiterState,
				Matcher:  request.{{ titleCase .Matcher }}WaiterMatch,
				{{- if .Argument }}Argument: "{{ .Argument }}",{{ end }}
				Expected: {{ .ExpectedString }},
			},
			{{ end }}
		},
		Logger: c.Config.Logger,
		NewRequest: func(opts []request.Option) (*request.Request, error) {
			var inCpy {{ .Operation.InputRef.GoType }}
			if input != nil  {
				tmp := *input
				inCpy = &tmp
			}
			req, _ := c.{{ .OperationName }}Request(inCpy)
			req.SetContext(ctx)
			req.ApplyOptions(opts...)
			return req, nil
		},
	}
	w.ApplyOptions(opts...)

	return w.WaitWithContext(ctx)
}
{{- end }}

{{ define "waiter interface" }}
WaitUntil{{ .Name }}({{ .Operation.InputRef.GoTypeWithPkgName }}) error
WaitUntil{{ .Name }}WithContext(aws.Context, {{ .Operation.InputRef.GoTypeWithPkgName }}, ...request.WaiterOption) error
{{- end }}
`))

// InterfaceSignature returns a string representing the Waiter's interface
// function signature.
func (w *Waiter) InterfaceSignature() string {
	var buf bytes.Buffer
	if err := waiterTmpls.ExecuteTemplate(&buf, "waiter interface", w); err != nil {
		panic(err)
	}

	return strings.TrimSpace(buf.String())
}

// GoCode returns the generated Go code for an individual waiter.
func (w *Waiter) GoCode() string {
	var buf bytes.Buffer
	if err := waiterTmpls.ExecuteTemplate(&buf, "waiter", w); err != nil {
		panic(err)
	}

	return buf.String()
}
