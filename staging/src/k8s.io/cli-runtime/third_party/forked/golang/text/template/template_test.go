package template

import (
	"io"
	"strings"
	"testing"
)

type data struct {
	Name string
	Tags map[string]int
	Kids []data
}

func (d data) Method() string { return "m" }

func TestExecute(t *testing.T) {
	in := data{Name: "a", Tags: map[string]int{"y": 2, "x": 1}, Kids: []data{{Name: "b"}}}
	funcs := FuncMap{"upper": strings.ToUpper, "method": data.Method}
	for _, tc := range []struct{ tmpl, want string }{
		{`{{.Name}}`, "a"},
		{`{{.Tags.x}}{{index .Tags "y"}}`, "12"},
		{`{{range $k, $v := .Tags}}{{$k}}={{$v}} {{end}}`, "x=1 y=2 "},
		{`{{range .Kids}}{{.Name}}{{end}}{{len .Kids}}`, "b1"},
		{`{{with .Kids}}{{(index . 0).Name}}{{else}}none{{end}}`, "b"},
		{`{{if and (eq .Name "a") (not .Tags.z)}}yes{{end}}`, "yes"},
		{`{{.Name | upper | printf "%s!"}}`, "A!"},
		{`{{define "x"}}[{{.}}]{{end}}{{template "x" .Name}}`, "[a]"},
		{`{{method .}}`, "m"},
	} {
		var b strings.Builder
		if err := Must(New("t").Funcs(funcs).Parse(tc.tmpl)).Execute(&b, in); err != nil || b.String() != tc.want {
			t.Errorf("%s: got %q, %v; want %q", tc.tmpl, b.String(), err, tc.want)
		}
	}
}

func TestExecuteTemplate(t *testing.T) {
	tmpl := Must(New("t").Parse(`{{define "a"}}A{{.}}{{end}}`))
	var b strings.Builder
	if err := tmpl.ExecuteTemplate(&b, "a", 1); err != nil || b.String() != "A1" || tmpl.Lookup("a") == nil {
		t.Fatalf("got %q, %v", b.String(), err)
	}
}

func TestMissingKeyOption(t *testing.T) {
	err := Must(New("t").Option("missingkey=error").Parse(`{{.x}}`)).Execute(io.Discard, map[string]int{})
	if err == nil || !strings.Contains(err.Error(), `map has no entry for key "x"`) {
		t.Fatalf("got %v", err)
	}
}

// A name never resolves to a method; register a function instead.
func TestNoMethodResolution(t *testing.T) {
	err := Must(New("t").Parse(`{{.Method}}`)).Execute(io.Discard, data{})
	if err == nil || !strings.Contains(err.Error(), "can't evaluate field Method") {
		t.Fatalf("want field error, got %v", err)
	}
}
