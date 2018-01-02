// +build codegen

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"os"
	"regexp"
	"strings"

	xhtml "golang.org/x/net/html"
)

type apiDocumentation struct {
	*API
	Operations map[string]string
	Service    string
	Shapes     map[string]shapeDocumentation
}

type shapeDocumentation struct {
	Base string
	Refs map[string]string
}

// AttachDocs attaches documentation from a JSON filename.
func (a *API) AttachDocs(filename string) {
	d := apiDocumentation{API: a}

	f, err := os.Open(filename)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	err = json.NewDecoder(f).Decode(&d)
	if err != nil {
		panic(err)
	}

	d.setup()

}

func (d *apiDocumentation) setup() {
	d.API.Documentation = docstring(d.Service)

	for op, doc := range d.Operations {
		d.API.Operations[op].Documentation = docstring(doc)
	}

	for shape, info := range d.Shapes {
		if sh := d.API.Shapes[shape]; sh != nil {
			sh.Documentation = docstring(info.Base)
		}

		for ref, doc := range info.Refs {
			if doc == "" {
				continue
			}

			parts := strings.Split(ref, "$")
			if len(parts) != 2 {
				fmt.Fprintf(os.Stderr, "Shape Doc %s has unexpected reference format, %q\n", shape, ref)
				continue
			}
			if sh := d.API.Shapes[parts[0]]; sh != nil {
				if m := sh.MemberRefs[parts[1]]; m != nil {
					m.Documentation = docstring(doc)
				}
			}
		}
	}
}

var reNewline = regexp.MustCompile(`\r?\n`)
var reMultiSpace = regexp.MustCompile(`\s+`)
var reComments = regexp.MustCompile(`<!--.*?-->`)
var reFullnameBlock = regexp.MustCompile(`<fullname>(.+?)<\/fullname>`)
var reFullname = regexp.MustCompile(`<fullname>(.*?)</fullname>`)
var reExamples = regexp.MustCompile(`<examples?>.+?<\/examples?>`)
var reEndNL = regexp.MustCompile(`\n+$`)

// docstring rewrites a string to insert godocs formatting.
func docstring(doc string) string {
	doc = strings.TrimSpace(doc)
	if doc == "" {
		return ""
	}

	doc = reNewline.ReplaceAllString(doc, "")
	doc = reMultiSpace.ReplaceAllString(doc, " ")
	doc = reComments.ReplaceAllString(doc, "")

	var fullname string
	parts := reFullnameBlock.FindStringSubmatch(doc)
	if len(parts) > 1 {
		fullname = parts[1]
	}
	// Remove full name block from doc string
	doc = reFullname.ReplaceAllString(doc, "")

	doc = reExamples.ReplaceAllString(doc, "")
	doc = generateDoc(doc)
	doc = reEndNL.ReplaceAllString(doc, "")
	doc = html.UnescapeString(doc)

	// Replace doc with full name if doc is empty.
	doc = strings.TrimSpace(doc)
	if len(doc) == 0 {
		doc = fullname
	}

	return commentify(doc)
}

const (
	indent = "   "
)

// style is what we want to prefix a string with.
// For instance, <li>Foo</li><li>Bar</li>, will generate
//    * Foo
//    * Bar
var style = map[string]string{
	"ul":   indent + "* ",
	"li":   indent + "* ",
	"code": indent,
	"pre":  indent,
}

// commentify converts a string to a Go comment
func commentify(doc string) string {
	if len(doc) == 0 {
		return ""
	}

	lines := strings.Split(doc, "\n")
	out := make([]string, 0, len(lines))
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		if i > 0 && line == "" && lines[i-1] == "" {
			continue
		}
		out = append(out, line)
	}

	if len(out) > 0 {
		out[0] = "// " + out[0]
		return strings.Join(out, "\n// ")
	}
	return ""
}

// wrap returns a rewritten version of text to have line breaks
// at approximately length characters. Line breaks will only be
// inserted into whitespace.
func wrap(text string, length int, isIndented bool) string {
	var buf bytes.Buffer
	var last rune
	var lastNL bool
	var col int

	for _, c := range text {
		switch c {
		case '\r': // ignore this
			continue // and also don't track `last`
		case '\n': // ignore this too, but reset col
			if col >= length || last == '\n' {
				buf.WriteString("\n")
			}
			buf.WriteString("\n")
			col = 0
		case ' ', '\t': // opportunity to split
			if col >= length {
				buf.WriteByte('\n')
				col = 0
				if isIndented {
					buf.WriteString(indent)
					col += 3
				}
			} else {
				// We only want to write a leading space if the col is greater than zero.
				// This will provide the proper spacing for documentation.
				buf.WriteRune(c)
				col++ // count column
			}
		default:
			buf.WriteRune(c)
			col++
		}
		lastNL = c == '\n'
		_ = lastNL
		last = c
	}
	return buf.String()
}

type tagInfo struct {
	tag        string
	key        string
	val        string
	txt        string
	raw        string
	closingTag bool
}

// generateDoc will generate the proper doc string for html encoded or plain text doc entries.
func generateDoc(htmlSrc string) string {
	tokenizer := xhtml.NewTokenizer(strings.NewReader(htmlSrc))
	tokens := buildTokenArray(tokenizer)
	scopes := findScopes(tokens)
	return walk(scopes)
}

func buildTokenArray(tokenizer *xhtml.Tokenizer) []tagInfo {
	tokens := []tagInfo{}
	for tt := tokenizer.Next(); tt != xhtml.ErrorToken; tt = tokenizer.Next() {
		switch tt {
		case xhtml.TextToken:
			txt := string(tokenizer.Text())
			if len(tokens) == 0 {
				info := tagInfo{
					raw: txt,
				}
				tokens = append(tokens, info)
			}
			tn, _ := tokenizer.TagName()
			key, val, _ := tokenizer.TagAttr()
			info := tagInfo{
				tag: string(tn),
				key: string(key),
				val: string(val),
				txt: txt,
			}
			tokens = append(tokens, info)
		case xhtml.StartTagToken:
			tn, _ := tokenizer.TagName()
			key, val, _ := tokenizer.TagAttr()
			info := tagInfo{
				tag: string(tn),
				key: string(key),
				val: string(val),
			}
			tokens = append(tokens, info)
		case xhtml.SelfClosingTagToken, xhtml.EndTagToken:
			tn, _ := tokenizer.TagName()
			key, val, _ := tokenizer.TagAttr()
			info := tagInfo{
				tag:        string(tn),
				key:        string(key),
				val:        string(val),
				closingTag: true,
			}
			tokens = append(tokens, info)
		}
	}
	return tokens
}

// walk is used to traverse each scoped block. These scoped
// blocks will act as blocked text where we do most of our
// text manipulation.
func walk(scopes [][]tagInfo) string {
	doc := ""
	// Documentation will be chunked by scopes.
	// Meaning, for each scope will be divided by one or more newlines.
	for _, scope := range scopes {
		indentStr, isIndented := priorityIndentation(scope)
		block := ""
		href := ""
		after := false
		level := 0
		lastTag := ""
		for _, token := range scope {
			if token.closingTag {
				endl := closeTag(token, level)
				block += endl
				level--
				lastTag = ""
			} else if token.txt == "" {
				if token.val != "" {
					href, after = formatText(token, "")
				}
				if level == 1 && isIndented {
					block += indentStr
				}
				level++
				lastTag = token.tag
			} else {
				if token.txt != " " {
					str, _ := formatText(token, lastTag)
					block += str
					if after {
						block += href
						after = false
					}
				} else {
					fmt.Println(token.tag)
					str, _ := formatText(tagInfo{}, lastTag)
					block += str
				}
			}
		}
		if !isIndented {
			block = strings.TrimPrefix(block, " ")
		}
		block = wrap(block, 72, isIndented)
		doc += block
	}
	return doc
}

// closeTag will divide up the blocks of documentation to be formated properly.
func closeTag(token tagInfo, level int) string {
	switch token.tag {
	case "pre", "li", "div":
		return "\n"
	case "p", "h1", "h2", "h3", "h4", "h5", "h6":
		return "\n\n"
	case "code":
		// indented code is only at the 0th level.
		if level == 0 {
			return "\n"
		}
	}
	return ""
}

// formatText will format any sort of text based off of a tag. It will also return
// a boolean to add the string after the text token.
func formatText(token tagInfo, lastTag string) (string, bool) {
	switch token.tag {
	case "a":
		if token.val != "" {
			return fmt.Sprintf(" (%s)", token.val), true
		}
	}

	// We don't care about a single space nor no text.
	if len(token.txt) == 0 || token.txt == " " {
		return "", false
	}

	// Here we want to indent code blocks that are newlines
	if lastTag == "code" {
		// Greater than one, because we don't care about newlines in the beginning
		block := ""
		if lines := strings.Split(token.txt, "\n"); len(lines) > 1 {
			for _, line := range lines {
				block += indent + line
			}
			block += "\n"
			return block, false
		}
	}
	return token.txt, false
}

// This is a parser to check what type of indention is needed.
func priorityIndentation(blocks []tagInfo) (string, bool) {
	if len(blocks) == 0 {
		return "", false
	}

	v, ok := style[blocks[0].tag]
	return v, ok
}

// Divides into scopes based off levels.
// For instance,
// <p>Testing<code>123</code></p><ul><li>Foo</li></ul>
// This has 2 scopes, the <p> and <ul>
func findScopes(tokens []tagInfo) [][]tagInfo {
	level := 0
	scope := []tagInfo{}
	scopes := [][]tagInfo{}
	for _, token := range tokens {
		// we will clear empty tagged tokens from the array
		txt := strings.TrimSpace(token.txt)
		tag := strings.TrimSpace(token.tag)
		if len(txt) == 0 && len(tag) == 0 {
			continue
		}

		scope = append(scope, token)

		// If it is a closing tag then we check what level
		// we are on. If it is 0, then that means we have found a
		// scoped block.
		if token.closingTag {
			level--
			if level == 0 {
				scopes = append(scopes, scope)
				scope = []tagInfo{}
			}
			// Check opening tags and increment the level
		} else if token.txt == "" {
			level++
		}
	}
	// In this case, we did not run into a closing tag. This would mean
	// we have plaintext for documentation.
	if len(scopes) == 0 {
		scopes = append(scopes, scope)
	}
	return scopes
}
