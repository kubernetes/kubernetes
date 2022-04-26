////////////////////////////////////////////////////////////////////////////
// Porgram: xmlfmt.go
// Purpose: Go XML Beautify from XML string using pure string manipulation
// Authors: Antonio Sun (c) 2016-2019, All rights reserved
////////////////////////////////////////////////////////////////////////////

package xmlfmt

import (
	"regexp"
	"strings"
)

var (
	reg = regexp.MustCompile(`<([/!]?)([^>]+?)(/?)>`)
	// NL is the newline string used in XML output, define for DOS-convenient.
	NL = "\r\n"
)

// FormatXML will (purly) reformat the XML string in a readable way, without any rewriting/altering the structure
func FormatXML(xmls, prefix, indent string) string {
	src := regexp.MustCompile(`(?s)>\s+<`).ReplaceAllString(xmls, "><")

	rf := replaceTag(prefix, indent)
	return (prefix + reg.ReplaceAllStringFunc(src, rf))
}

// replaceTag returns a closure function to do 's/(?<=>)\s+(?=<)//g; s(<(/?)([^>]+?)(/?)>)($indent+=$3?0:$1?-1:1;"<$1$2$3>"."\n".("  "x$indent))ge' as in Perl
// and deal with comments as well
func replaceTag(prefix, indent string) func(string) string {
	indentLevel := 0
	return func(m string) string {
		// head elem
		if strings.HasPrefix(m, "<?xml") {
			return NL + prefix + strings.Repeat(indent, indentLevel) + m
		}
		// empty elem
		if strings.HasSuffix(m, "/>") {
			return NL + prefix + strings.Repeat(indent, indentLevel) + m
		}
		// comment elem
		if strings.HasPrefix(m, "<!") {
			return NL + prefix + strings.Repeat(indent, indentLevel) + m
		}
		// end elem
		if strings.HasPrefix(m, "</") {
			indentLevel--
			return NL + prefix + strings.Repeat(indent, indentLevel) + m
		}
		defer func() {
			indentLevel++
		}()

		return NL + prefix + strings.Repeat(indent, indentLevel) + m
	}
}
