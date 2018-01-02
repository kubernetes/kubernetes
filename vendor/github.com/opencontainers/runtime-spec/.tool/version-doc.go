// +build ignore

package main

import (
	"fmt"
	"html/template"
	"os"

	"github.com/opencontainers/runtime-spec/specs-go"
)

var markdownTemplateString = `

**Specification Version:** *{{.}}*

`

var markdownTemplate = template.Must(template.New("markdown").Parse(markdownTemplateString))

func main() {
	if err := markdownTemplate.Execute(os.Stdout, specs.Version); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}
