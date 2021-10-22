// +build codegen

package api

import (
	"fmt"

	"github.com/aws/aws-sdk-go/private/protocol"
)

type examplesBuilder interface {
	BuildShape(*ShapeRef, map[string]interface{}, bool) string
	BuildList(string, string, *ShapeRef, []interface{}) string
	BuildComplex(string, string, *ShapeRef, *Shape, map[string]interface{}) string
	GoType(*ShapeRef, bool) string
	Imports(*API) string
}

type defaultExamplesBuilder struct {
	ShapeValueBuilder
}

// NewExamplesBuilder returns an initialized example builder for generating
// example input API shapes from a model.
func NewExamplesBuilder() defaultExamplesBuilder {
	b := defaultExamplesBuilder{
		ShapeValueBuilder: NewShapeValueBuilder(),
	}
	b.ParseTimeString = parseExampleTimeString
	return b
}

func (builder defaultExamplesBuilder) Imports(a *API) string {
	return `"fmt"
	"strings"
	"time"

	"` + SDKImportRoot + `/aws"
	"` + SDKImportRoot + `/aws/awserr"
	"` + SDKImportRoot + `/aws/session"
	"` + a.ImportPath() + `"
	`
}

// Returns a string which assigns the value of a time member by calling
// parseTime function defined in the file
func parseExampleTimeString(ref *ShapeRef, memName, v string) string {
	if ref.Location == "header" {
		return fmt.Sprintf("%s: parseTime(%q, %q),\n", memName, protocol.RFC822TimeFormat, v)
	}

	switch ref.API.Metadata.Protocol {
	case "json", "rest-json", "rest-xml", "ec2", "query":
		return fmt.Sprintf("%s: parseTime(%q, %q),\n", memName, protocol.ISO8601TimeFormat, v)
	default:
		panic("Unsupported time type: " + ref.API.Metadata.Protocol)
	}
}
