/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package kyaml provides an encoder for KYAML, a strict subset of YAML that is
// designed to be explicit and unambiguous.  KYAML is YAML, so any YAML parser
// should be able to read it.
//
// KYAML is designed to be halfway between YAML and JSON, with the following
// properties:
//   - Not whitespace-sensitive
//   - Allows comments
//   - Allows trailing commas
//   - Does not require quoted keys
//
// KYAML is an output format, and will follow these conventions:
//   - Always double-quote strings, even if they are not ambiguous.
//   - Only quote keys that might be ambiguously interpreted (e.g. "no" is
//     always quoted).
//   - Always use `{}` for structs and maps, and `[]` for lists.
//   - Economize on vertical space by cuddling some kinds of brackets together.
//   - Render multi-line strings with YAML's line folding, which is close to
//     the Go string literal format.
//
// KYAML also includes a document-separator "header" (still valid YAML), which
// helps to disambiguate a KYAML document from an ill-formed JSON document.
//
// Because KYAML is YAML, a KYAML multi-document is a YAML multi-document.
package kyaml

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	yaml "go.yaml.in/yaml/v3"
)

// Encoder formats objects or YAML data (JSON is valid YAML) into KYAML. KYAML
// is halfway between YAML and JSON, but is a strict subset of YAML, so it
// should should be readable by any YAML parser. It is designed to be explicit
// and unambiguous, and eschews significant whitespace.
type Encoder struct {
	// Compact tells the encoder to use compact formatting. This puts all the
	// data on one line, with no extra newlines, no comments, and no multi-line
	// formatting.
	Compact bool
}

// FromYAML renders a KYAML (multi-)document from YAML bytes (JSON is YAML),
// including the KYAML header. The result always has a trailing newline.
func (ky *Encoder) FromYAML(in io.Reader, out io.Writer) error {
	// We need a YAML decoder to handle multi-document streams.
	dec := yaml.NewDecoder(in)

	// Process each document in the stream.
	for {
		var doc yaml.Node
		err := dec.Decode(&doc)
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error decoding: %v", err)
		}
		if doc.Kind != yaml.DocumentNode {
			return fmt.Errorf("kyaml internal error: line %d: expected a document node, got %s", doc.Line, ky.nodeKindString(doc.Kind))
		}

		// Always emit a document separator, which helps disambiguate between YAML
		// and JSON.
		if _, err := fmt.Fprintln(out, "---"); err != nil {
			return err
		}

		if err := ky.renderDocument(&doc, 0, ky.flags(), out); err != nil {
			return err
		}
		fmt.Fprintf(out, "\n")
	}

	return nil
}

// FromObject renders a KYAML document from a Go object, including the KYAML
// header. The result always has a trailing newline.
func (ky *Encoder) FromObject(obj any, out io.Writer) error {
	jb, err := json.Marshal(obj)
	if err != nil {
		return fmt.Errorf("error marshaling to JSON: %v", err)
	}
	// JSON is YAML.
	return ky.FromYAML(bytes.NewReader(jb), out)
}

// Marshal renders a single Go object as KYAML, without the header or trailing
// newline.
func (ky *Encoder) Marshal(obj any) ([]byte, error) {
	// Convert the object to JSON bytes to take advantage of all the JSON tag
	// handling and things like that.
	jb, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("error marshaling to JSON: %v", err)
	}

	buf := &bytes.Buffer{}
	// JSON is YAML.
	if err := ky.fromObjectYAML(bytes.NewReader(jb), buf); err != nil {
		return nil, fmt.Errorf("error rendering object: %v", err)
	}
	return buf.Bytes(), nil
}

func (ky *Encoder) fromObjectYAML(in io.Reader, out io.Writer) error {
	yb, err := io.ReadAll(in)
	if err != nil {
		return err
	}

	var doc yaml.Node
	if err := yaml.Unmarshal(yb, &doc); err != nil {
		return fmt.Errorf("error decoding: %v", err)
	}
	if doc.Kind != yaml.DocumentNode {
		return fmt.Errorf("kyaml internal error: line %d: expected document node, got %s", doc.Line, ky.nodeKindString(doc.Kind))
	}

	if err := ky.renderNode(&doc, 0, ky.flags(), out); err != nil {
		return fmt.Errorf("error rendering document: %v", err)
	}

	return nil
}

// From the YAML spec.
const (
	intTag       = "!!int"
	floatTag     = "!!float"
	boolTag      = "!!bool"
	strTag       = "!!str"
	timestampTag = "!!timestamp"
	seqTag       = "!!seq"
	mapTag       = "!!map"
	nullTag      = "!!null"
	binaryTag    = "!!binary"
	mergeTag     = "!!merge"
)

type flagMask uint64

const (
	flagsNone     flagMask = 0
	flagLazyQuote flagMask = 0x01
	flagCompact   flagMask = 0x02
)

// flags returns a flagMask representing the current encoding options. It can
// be used directly or OR'ed with another mask.
func (ky *Encoder) flags() flagMask {
	flags := flagsNone
	if ky.Compact {
		flags |= flagCompact
	}
	return flags
}

// renderNode processes a YAML node, calling the appropriate render function
// for its type.  Each render function should assume that the output "cursor"
// is positioned at the start of the node and should not emit a final newline.
// If a render function needs to linewrap or indent (e.g. a struct), it should
// assume the indent level is currently correct for the node type itself, and
// may need to indent more.
func (ky *Encoder) renderNode(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	if node == nil {
		return nil
	}

	switch node.Kind {
	case yaml.DocumentNode:
		return ky.renderDocument(node, indent, flags, out)
	case yaml.ScalarNode:
		return ky.renderScalar(node, indent, flags, out)
	case yaml.SequenceNode:
		return ky.renderSequence(node, indent, flags, out)
	case yaml.MappingNode:
		return ky.renderMapping(node, indent, flags, out)
	case yaml.AliasNode:
		return ky.renderAlias(node, indent, flags, out)
	}
	return fmt.Errorf("kyaml internal error: line %d: unknown node kind %v", node.Line, node.Kind)
}

// renderDocument processes a YAML document node, rendering it to the output.
// This function assumes that the output "cursor" is positioned at the start of
// the document. This does not emit a final newline.
func (ky *Encoder) renderDocument(doc *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	if len(doc.Content) == 0 {
		return fmt.Errorf("kyaml internal error: line %d: document has no content node (%d)", doc.Line, len(doc.Content))
	}
	if len(doc.Content) > 1 {
		return fmt.Errorf("kyaml internal error: line %d: document has more than one content node (%d)", doc.Line, len(doc.Content))
	}
	if indent != 0 {
		return fmt.Errorf("kyaml internal error: line %d: document non-zero indent (%d)", doc.Line, indent)
	}

	compact := flags&flagCompact != 0

	// For document nodes, the cursor is assumed to be ready to render.
	child := doc.Content[0]
	if !compact {
		if len(doc.HeadComment) > 0 {
			ky.renderComments(doc.HeadComment, indent, out)
			fmt.Fprint(out, "\n")
		}
		if len(child.HeadComment) > 0 {
			ky.renderComments(child.HeadComment, indent, out)
			fmt.Fprint(out, "\n")
		}
	}
	if err := ky.renderNode(child, indent, flags, out); err != nil {
		return err
	}
	if !compact {
		if len(child.LineComment) > 0 {
			ky.renderComments(" "+child.LineComment, 0, out)
		}
		if len(child.FootComment) > 0 {
			fmt.Fprint(out, "\n")
			ky.renderComments(child.FootComment, indent, out)
		}
		if len(doc.LineComment) > 0 {
			fmt.Fprint(out, "\n")
			ky.renderComments(" "+doc.LineComment, 0, out)
		}
		if len(doc.FootComment) > 0 {
			fmt.Fprint(out, "\n")
			ky.renderComments(doc.FootComment, indent, out)
		}
	}
	return nil
}

// renderScalar processes a YAML scalar node, rendering it to the output.  This
// DOES NOT render a trailing newline or head/line/foot comments, as those
// require the parent context.
func (ky *Encoder) renderScalar(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	switch node.Tag {
	case intTag, floatTag, boolTag, nullTag:
		fmt.Fprint(out, node.Value)
	case strTag, timestampTag:
		return ky.renderString(node.Value, indent+1, flags, out)
	default:
		return fmt.Errorf("kyaml internal error: line %d: unknown tag %q on scalar node %q", node.Line, node.Tag, node.Value)
	}
	return nil
}

const kyamlFoldStr = "\\\n"

var regularEscapeMap = map[rune]string{
	'\n': "\\n" + kyamlFoldStr, // use YAML's line folding to make the output more readable
	'\t': "\t",                 // literal tab
}
var compactEscapeMap = map[rune]string{
	'\n': "\\n",
	'\t': "\\t",
}

// renderString processes a string (either single-line or multi-line),
// rendering it to the output.  This DOES NOT render a trailing newline.
func (ky *Encoder) renderString(val string, indent int, flags flagMask, out io.Writer) error {
	lazyQuote := flags&flagLazyQuote != 0
	compact := flags&flagCompact != 0
	multi := strings.Contains(val, "\n")

	if !multi && lazyQuote && !needsQuotes(val) {
		fmt.Fprint(out, val)
		return nil
	}

	// Special cases for certain input.
	escapeOverrides := regularEscapeMap
	if compact {
		escapeOverrides = compactEscapeMap
	}

	//
	// The rest of this is borrowed from Go's strconv.Quote implementation.
	//

	// accumulate into a buffer
	buf := &bytes.Buffer{}

	// opening quote
	fmt.Fprint(buf, `"`)
	if multi && !compact {
		fmt.Fprint(buf, kyamlFoldStr)
	}

	// Iterating a string with invalid UTF8 returns RuneError rather than the
	// bytes, so we iterate the string and decode the runes. This is a bit
	// slower, but gives us a better result.
	s := val
	for width := 0; len(s) > 0; s = s[width:] {
		r := rune(s[0])
		width = 1
		if r >= utf8.RuneSelf {
			r, width = utf8.DecodeRuneInString(s)
		}
		if width == 1 && r == utf8.RuneError {
			fmt.Fprint(buf, `\x`)
			fmt.Fprintf(buf, "%02x", s[0])
			continue
		}
		ky.appendEscapedRune(r, indent, escapeOverrides, buf)
	}

	// closing quote
	afterNewline := buf.Bytes()[len(buf.Bytes())-1] == '\n'
	if multi && !compact {
		if !afterNewline {
			fmt.Fprint(buf, kyamlFoldStr)
		}
		ky.writeIndent(indent, buf)
	}
	fmt.Fprint(buf, `"`)

	fmt.Fprint(out, buf.String())

	return nil
}

var allowedUnquotedAnywhere = map[rune]bool{
	'_': true,
}

var allowedUnquotedInterior = map[rune]bool{
	'-': true,
	'.': true,
	'/': true,
}

func needsQuotes(s string) bool {
	if s == "" {
		return true
	}
	if isTypeAmbiguous(s) {
		return true
	}
	runes := []rune(s)
	for i, r := range runes {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || allowedUnquotedAnywhere[r] {
			continue
		}
		if i > 0 && i < len(runes)-1 && allowedUnquotedInterior[r] {
			continue
		}
		// it's something we don't explicitly allow
		return true
	}
	return false
}

// From https://yaml.org/type/int.html and https://yaml.org/type/float.html
var sexagesimalRE = regexp.MustCompile(`^[+-]?[1-9][0-9_]*(:[0-5]?[0-9])+(\.[0-9_]*)?$`)

// isTypeAmbiguous returns true if a YAML parser might interpret the unquoted
// form of the string argument as a YAML type other than string (e.g. `true`
// would be interpreted as a boolean).
func isTypeAmbiguous(s string) bool {
	// Null-like strings: https://yaml.org/type/null.html
	if len(s) <= 5 {
		switch strings.ToLower(s) {
		case "null", "~", "":
			return true
		}
	}

	// Boolean-like strings: https://yaml.org/type/bool.html
	if _, err := strconv.ParseBool(s); err == nil {
		return true
	}
	if len(s) <= 5 {
		switch strings.ToLower(s) {
		case "true", "y", "yes", "on", "false", "n", "no", "off":
			return true
		}
	}

	// Number-like strings: https://yaml.org/type/int.html and
	// https://yaml.org/type/float.html
	//
	// NOTE: the stripping of underscores is gross.
	sWithoutUnderscores := strings.ReplaceAll(s, "_", "")
	// Handles binary ("0b"), octal ("0" or "0o"), decimal, and hex ("0x")
	if _, err := strconv.ParseInt(sWithoutUnderscores, 0, 64); err == nil && !isSyntaxError(err) {
		return true
	}
	// Handles standard and scientific notation.
	if _, err := strconv.ParseFloat(sWithoutUnderscores, 64); err == nil && !isSyntaxError(err) {
		return true
	}

	// Sexagesimal strings like "11:00" (in YAML 1.1, removed in 1.2):
	// https://yaml.org/type/int.html and https://yaml.org/type/float.html
	if sexagesimalRE.MatchString(s) {
		return true
	}

	// Infinity and NaN: https://yaml.org/type/float.html
	if len(s) <= 5 {
		switch strings.ToLower(s) {
		case ".inf", "-.inf", "+.inf", ".nan":
			return true
		}
	}

	// Time-like strings
	if _, matches := parseTimestamp(s); matches {
		return true
	}

	return false
}

func isSyntaxError(err error) bool {
	var numerr *strconv.NumError
	if ok := errors.As(err, &numerr); ok {
		return errors.Is(numerr.Err, strconv.ErrSyntax)
	}
	return false
}

// This is a subset of the formats allowed by the regular expression
// defined at http://yaml.org/type/timestamp.html.
//
// NOTE: This was copied from go.yaml.in/yaml/v2
var allowedTimestampFormats = []string{
	"2006-1-2T15:4:5.999999999Z07:00", // RCF3339Nano with short date fields.
	"2006-1-2t15:4:5.999999999Z07:00", // RFC3339Nano with short date fields and lower-case "t".
	"2006-1-2 15:4:5.999999999",       // space separated with no time zone
	"2006-1-2",                        // date only
	// Notable exception: time.Parse cannot handle: "2001-12-14 21:59:43.10 -5"
	// from the set of examples.
}

// parseTimestamp parses s as a timestamp string and
// returns the timestamp and reports whether it succeeded.
// Timestamp formats are defined at http://yaml.org/type/timestamp.html
//
// NOTE: This was copied from go.yaml.in/yaml/v2
func parseTimestamp(s string) (time.Time, bool) {
	// TODO write code to check all the formats supported by
	// http://yaml.org/type/timestamp.html instead of using time.Parse.

	// Quick check: all date formats start with YYYY-.
	i := 0
	for ; i < len(s); i++ {
		if c := s[i]; c < '0' || c > '9' {
			break
		}
	}
	if i != 4 || i == len(s) || s[i] != '-' {
		return time.Time{}, false
	}
	for _, format := range allowedTimestampFormats {
		if t, err := time.Parse(format, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// We use a buffer here so we can peek backwards.
func (ky *Encoder) appendEscapedRune(r rune, indent int, escapeOverrides map[rune]string, buf *bytes.Buffer) {
	afterNewline := buf.Bytes()[len(buf.Bytes())-1] == '\n'

	if afterNewline {
		ky.writeIndent(indent, buf)
		// We want to preserve leading whitespace in the source string, so if
		// we find whitespace, we need to escape it.  We don't want to
		// escape lines without leading whitespace, but we DO want to render
		// the result with fidelity to vertical alignment, so we write an extra
		// space.  This is OK, because all whitespace before the first
		// non-whitespace character is dropped, as per YAML spec. If there are
		// no lines with leading whitespace it looks like the indent is one too
		// many, which seems OK.
		if unicode.IsSpace(r) && r != '\n' {
			buf.WriteRune('\\')
		} else {
			buf.WriteRune(' ')
		}
	}
	if s, found := escapeOverrides[r]; found {
		buf.WriteString(s)
		return
	}
	if r == '"' || r == '\\' { // always escaped
		buf.WriteRune('\\')
		buf.WriteRune(r)
		return
	}
	if unicode.IsPrint(r) {
		buf.WriteRune(r)
		return
	}
	switch r {
	case '\a':
		buf.WriteString(`\a`)
	case '\b':
		buf.WriteString(`\b`)
	case '\f':
		buf.WriteString(`\f`)
	case '\n':
		buf.WriteString(`\n`)
	case '\r':
		buf.WriteString(`\r`)
	case '\t':
		buf.WriteString(`\t`)
	case '\v':
		buf.WriteString(`\v`)
	case '\x00':
		buf.WriteString(`\0`)
	case '\x1b':
		buf.WriteString(`\e`)
	case '\x85':
		buf.WriteString(`\N`)
	case '\xa0':
		buf.WriteString(`\_`)
	case '\u2028':
		buf.WriteString(`\L`)
	case '\u2029':
		buf.WriteString(`\P`)
	default:
		const hexits = "0123456789abcdef"
		switch {
		case r < ' ' || r == 0x7f:
			buf.WriteString(`\x`)
			buf.WriteByte(hexits[byte(r)>>4])
			buf.WriteByte(hexits[byte(r)&0xF])
		case !utf8.ValidRune(r):
			r = utf8.RuneError
			fallthrough
		case r < 0x10000:
			buf.WriteString(`\u`)
			for s := 12; s >= 0; s -= 4 {
				buf.WriteByte(hexits[r>>uint(s)&0xF])
			}
		default:
			buf.WriteString(`\U`)
			for s := 28; s >= 0; s -= 4 {
				buf.WriteByte(hexits[r>>uint(s)&0xF])
			}
		}
	}
}

// renderSequence processes a YAML sequence node, rendering it to the output.  This
// DOES NOT render a trailing newline or head/line/foot comments of the sequence
// itself, but DOES render comments of the child nodes.
func (ky *Encoder) renderSequence(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	if len(node.Content) == 0 {
		fmt.Fprint(out, "[]")
		return nil
	}
	if flags&flagCompact != 0 {
		return ky.renderCompactSequence(node, flags, out)
	}

	// See if this list can use cuddled formatting.
	cuddle := true
	for _, child := range node.Content {
		if !isCuddledKind(child) {
			cuddle = false
			break
		}
		if len(child.HeadComment)+len(child.LineComment)+len(child.FootComment) > 0 {
			cuddle = false
			break
		}
	}

	if cuddle {
		return ky.renderCuddledSequence(node, indent, flags, out)
	}
	return ky.renderUncuddledSequence(node, indent, flags, out)
}

// renderCompactSequence renders a YAML sequence node in compact form.
func (ky *Encoder) renderCompactSequence(node *yaml.Node, flags flagMask, out io.Writer) error {
	fmt.Fprint(out, "[")
	for i, child := range node.Content {
		if i > 0 {
			fmt.Fprint(out, ", ")
		}
		if err := ky.renderNode(child, 0, flags, out); err != nil {
			return err
		}
	}
	fmt.Fprint(out, "]")
	return nil
}

// renderCuddledSequence processes a YAML sequence node which has already been
// determined to be cuddled.  We only cuddle sequences of structs or lists
// which have no comments.
func (ky *Encoder) renderCuddledSequence(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	fmt.Fprint(out, "[")
	for i, child := range node.Content {
		// Each iteration should leave us cuddled for the next item.
		if i > 0 {
			fmt.Fprint(out, ", ")
		}
		if err := ky.renderNode(child, indent, flags, out); err != nil {
			return err
		}
	}
	fmt.Fprint(out, "]")
	return nil
}

func (ky *Encoder) renderUncuddledSequence(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	// Get into the right state for the first item.
	fmt.Fprint(out, "[\n")
	ky.writeIndent(indent, out)
	for _, child := range node.Content {
		// Each iteration should leave us ready to close the list. Since we
		// have an item to render, we need 1 more indent.
		ky.writeIndent(1, out)

		if len(child.HeadComment) > 0 {
			ky.renderComments(child.HeadComment, indent+1, out)
			fmt.Fprint(out, "\n")
			ky.writeIndent(indent+1, out)
		}

		if err := ky.renderNode(child, indent+1, flags, out); err != nil {
			return err
		}

		fmt.Fprint(out, ",")
		if len(child.LineComment) > 0 {
			ky.renderComments(" "+child.LineComment, 0, out)
		}
		fmt.Fprint(out, "\n")
		ky.writeIndent(indent, out)
		if len(child.FootComment) > 0 {
			ky.writeIndent(1, out)
			ky.renderComments(child.FootComment, indent+1, out)
			fmt.Fprint(out, "\n")
			ky.writeIndent(indent, out)
		}
	}
	fmt.Fprint(out, "]")

	return nil
}

func (ky *Encoder) nodeKindString(kind yaml.Kind) string {
	switch kind {
	case yaml.DocumentNode:
		return "document"
	case yaml.ScalarNode:
		return "scalar"
	case yaml.MappingNode:
		return "mapping"
	case yaml.SequenceNode:
		return "sequence"
	case yaml.AliasNode:
		return "alias"
	default:
		return "unknown"
	}
}

func isCuddledKind(node *yaml.Node) bool {
	if node == nil {
		return false
	}
	switch node.Kind {
	case yaml.SequenceNode, yaml.MappingNode:
		return true
	case yaml.AliasNode:
		return isCuddledKind(node.Alias)
	}
	return false
}

// renderMapping processes a YAML mapping node, rendering it to the output.  This
// DOES NOT render a trailing newline or head/line/foot comments of the mapping
// itself, but DOES render comments of the child nodes.
func (ky *Encoder) renderMapping(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	if len(node.Content) == 0 {
		fmt.Fprint(out, "{}")
		return nil
	}

	if flags&flagCompact != 0 {
		return ky.renderCompactMapping(node, flags, out)
	}

	joinComments := func(a, b string) string {
		if len(a) > 0 && len(b) > 0 {
			return a + "\n" + b
		}
		return a + b
	}

	fmt.Fprint(out, "{\n")
	for i := 0; i < len(node.Content); i += 2 {
		key := node.Content[i]
		val := node.Content[i+1]

		ky.writeIndent(indent+1, out)

		// Only one of these should be set.
		if comments := joinComments(key.HeadComment, val.HeadComment); len(comments) > 0 {
			ky.renderComments(comments, indent+1, out)
			fmt.Fprint(out, "\n")
			ky.writeIndent(indent+1, out)
		}

		// Mapping keys are always strings in KYAML, even if the YAML node says
		// otherwise.
		if err := ky.renderString(key.Value, indent+1, flagLazyQuote|flagCompact, out); err != nil {
			return err
		}
		fmt.Fprint(out, ": ")
		if err := ky.renderNode(val, indent+1, flags, out); err != nil {
			return err
		}
		fmt.Fprint(out, ",")
		if len(key.LineComment) > 0 && len(val.LineComment) > 0 {
			return fmt.Errorf("kyaml internal error: line %d: both key and value have line comments", key.Line)
		}
		if len(key.LineComment) > 0 {
			ky.renderComments(" "+key.LineComment, 0, out)
		} else if len(val.LineComment) > 0 {
			ky.renderComments(" "+val.LineComment, 0, out)
		}
		fmt.Fprint(out, "\n")
		// Only one of these should be set.
		if comments := joinComments(key.FootComment, val.FootComment); len(comments) > 0 {
			ky.writeIndent(indent+1, out)
			ky.renderComments(comments, indent+1, out)
			fmt.Fprint(out, "\n")
		}
	}
	ky.writeIndent(indent, out)
	fmt.Fprint(out, "}")
	return nil
}

// renderCompactMapping renders a YAML mapping node in compact form.
func (ky *Encoder) renderCompactMapping(node *yaml.Node, flags flagMask, out io.Writer) error {
	fmt.Fprint(out, "{")
	for i := 0; i < len(node.Content); i += 2 {
		key := node.Content[i]
		val := node.Content[i+1]

		if i > 0 {
			fmt.Fprint(out, ", ")
		}
		// Mapping keys are always strings in KYAML, even if the YAML node says
		// otherwise.
		if err := ky.renderString(key.Value, 0, flags|flagLazyQuote|flagCompact, out); err != nil {
			return err
		}
		fmt.Fprint(out, ": ")
		if err := ky.renderNode(val, 0, flags, out); err != nil {
			return err
		}
	}
	fmt.Fprint(out, "}")
	return nil
}

func (ky *Encoder) writeIndent(level int, out io.Writer) {
	const indentString = "  "
	for range level {
		fmt.Fprint(out, indentString)
	}
}

// renderCommentBlock writes the comments node to the output.  This assumes the
// cursor is at the right place to start writing and DOES NOT render a trailing
// newline.
func (ky *Encoder) renderComments(comments string, indent int, out io.Writer) {
	if len(comments) == 0 {
		return
	}
	lines := strings.Split(comments, "\n")
	for i, line := range lines {
		if i > 0 {
			fmt.Fprint(out, "\n")
			ky.writeIndent(indent, out)
		}
		fmt.Fprint(out, line)
	}
}

func (ky *Encoder) renderAlias(node *yaml.Node, indent int, flags flagMask, out io.Writer) error {
	if node.Alias != nil {
		return ky.renderNode(node.Alias, indent+1, flags, out)
	}
	return nil
}
