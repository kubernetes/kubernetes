// registry-api-descriptor-template uses the APIDescriptor defined in the
// api/v2 package to execute templates passed to the command line.
//
// For example, to generate a new API specification, one would execute the
// following command from the repo root:
//
// 	$ registry-api-descriptor-template docs/spec/api.md.tmpl > docs/spec/api.md
//
// The templates are passed in the api/v2.APIDescriptor object. Please see the
// package documentation for fields available on that object. The template
// syntax is from Go's standard library text/template package. For information
// on Go's template syntax, please see golang.org/pkg/text/template.
package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"text/template"

	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
)

var spaceRegex = regexp.MustCompile(`\n\s*`)

func main() {

	if len(os.Args) != 2 {
		log.Fatalln("please specify a template to execute.")
	}

	path := os.Args[1]
	filename := filepath.Base(path)

	funcMap := template.FuncMap{
		"removenewlines": func(s string) string {
			return spaceRegex.ReplaceAllString(s, " ")
		},
		"statustext":    http.StatusText,
		"prettygorilla": prettyGorillaMuxPath,
	}

	tmpl := template.Must(template.New(filename).Funcs(funcMap).ParseFiles(path))

	data := struct {
		RouteDescriptors []v2.RouteDescriptor
		ErrorDescriptors []errcode.ErrorDescriptor
	}{
		RouteDescriptors: v2.APIDescriptor.RouteDescriptors,
		ErrorDescriptors: append(errcode.GetErrorCodeGroup("registry.api.v2"),
			// The following are part of the specification but provided by errcode default.
			errcode.ErrorCodeUnauthorized.Descriptor(),
			errcode.ErrorCodeDenied.Descriptor(),
			errcode.ErrorCodeUnsupported.Descriptor()),
	}

	if err := tmpl.Execute(os.Stdout, data); err != nil {
		log.Fatalln(err)
	}
}

// prettyGorillaMuxPath removes the regular expressions from a gorilla/mux
// route string, making it suitable for documentation.
func prettyGorillaMuxPath(s string) string {
	// Stateful parser that removes regular expressions from gorilla
	// routes. It correctly handles balanced bracket pairs.

	var output string
	var label string
	var level int

start:
	if s[0] == '{' {
		s = s[1:]
		level++
		goto capture
	}

	output += string(s[0])
	s = s[1:]

	goto end
capture:
	switch s[0] {
	case '{':
		level++
	case '}':
		level--

		if level == 0 {
			s = s[1:]
			goto label
		}
	case ':':
		s = s[1:]
		goto skip
	default:
		label += string(s[0])
	}
	s = s[1:]
	goto capture
skip:
	switch s[0] {
	case '{':
		level++
	case '}':
		level--
	}
	s = s[1:]

	if level == 0 {
		goto label
	}

	goto skip
label:
	if label != "" {
		output += "<" + label + ">"
		label = ""
	}
end:
	if s != "" {
		goto start
	}

	return output

}
