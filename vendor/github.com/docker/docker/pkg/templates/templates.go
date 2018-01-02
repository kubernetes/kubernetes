package templates

import (
	"bytes"
	"encoding/json"
	"strings"
	"text/template"
)

// basicFunctions are the set of initial
// functions provided to every template.
var basicFunctions = template.FuncMap{
	"json": func(v interface{}) string {
		buf := &bytes.Buffer{}
		enc := json.NewEncoder(buf)
		enc.SetEscapeHTML(false)
		enc.Encode(v)
		// Remove the trailing new line added by the encoder
		return strings.TrimSpace(buf.String())
	},
	"split":    strings.Split,
	"join":     strings.Join,
	"title":    strings.Title,
	"lower":    strings.ToLower,
	"upper":    strings.ToUpper,
	"pad":      padWithSpace,
	"truncate": truncateWithLength,
}

// HeaderFunctions are used to created headers of a table.
// This is a replacement of basicFunctions for header generation
// because we want the header to remain intact.
// Some functions like `split` are irrelevant so not added.
var HeaderFunctions = template.FuncMap{
	"json": func(v string) string {
		return v
	},
	"title": func(v string) string {
		return v
	},
	"lower": func(v string) string {
		return v
	},
	"upper": func(v string) string {
		return v
	},
	"truncate": func(v string, l int) string {
		return v
	},
}

// Parse creates a new anonymous template with the basic functions
// and parses the given format.
func Parse(format string) (*template.Template, error) {
	return NewParse("", format)
}

// NewParse creates a new tagged template with the basic functions
// and parses the given format.
func NewParse(tag, format string) (*template.Template, error) {
	return template.New(tag).Funcs(basicFunctions).Parse(format)
}

// padWithSpace adds whitespace to the input if the input is non-empty
func padWithSpace(source string, prefix, suffix int) string {
	if source == "" {
		return source
	}
	return strings.Repeat(" ", prefix) + source + strings.Repeat(" ", suffix)
}

// truncateWithLength truncates the source string up to the length provided by the input
func truncateWithLength(source string, length int) string {
	if len(source) < length {
		return source
	}
	return source[:length]
}
