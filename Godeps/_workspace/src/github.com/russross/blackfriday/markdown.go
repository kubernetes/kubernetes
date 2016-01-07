//
// Blackfriday Markdown Processor
// Available at http://github.com/russross/blackfriday
//
// Copyright © 2011 Russ Ross <russ@russross.com>.
// Distributed under the Simplified BSD License.
// See README.md for details.
//

//
//
// Markdown parsing and processing
//
//

// Blackfriday markdown processor.
//
// Translates plain text with simple formatting rules into HTML or LaTeX.
package blackfriday

import (
	"bytes"
	"unicode/utf8"
)

const VERSION = "1.1"

// These are the supported markdown parsing extensions.
// OR these values together to select multiple extensions.
const (
	EXTENSION_NO_INTRA_EMPHASIS          = 1 << iota // ignore emphasis markers inside words
	EXTENSION_TABLES                                 // render tables
	EXTENSION_FENCED_CODE                            // render fenced code blocks
	EXTENSION_AUTOLINK                               // detect embedded URLs that are not explicitly marked
	EXTENSION_STRIKETHROUGH                          // strikethrough text using ~~test~~
	EXTENSION_LAX_HTML_BLOCKS                        // loosen up HTML block parsing rules
	EXTENSION_SPACE_HEADERS                          // be strict about prefix header rules
	EXTENSION_HARD_LINE_BREAK                        // translate newlines into line breaks
	EXTENSION_TAB_SIZE_EIGHT                         // expand tabs to eight spaces instead of four
	EXTENSION_FOOTNOTES                              // Pandoc-style footnotes
	EXTENSION_NO_EMPTY_LINE_BEFORE_BLOCK             // No need to insert an empty line to start a (code, quote, order list, unorder list)block
	EXTENSION_HEADER_IDS                             // specify header IDs  with {#id}
	EXTENSION_TITLEBLOCK                             // Titleblock ala pandoc
	EXTENSION_AUTO_HEADER_IDS                        // Create the header ID from the text

	commonHtmlFlags = 0 |
		HTML_USE_XHTML |
		HTML_USE_SMARTYPANTS |
		HTML_SMARTYPANTS_FRACTIONS |
		HTML_SMARTYPANTS_LATEX_DASHES

	commonExtensions = 0 |
		EXTENSION_NO_INTRA_EMPHASIS |
		EXTENSION_TABLES |
		EXTENSION_FENCED_CODE |
		EXTENSION_AUTOLINK |
		EXTENSION_STRIKETHROUGH |
		EXTENSION_SPACE_HEADERS |
		EXTENSION_HEADER_IDS
)

// These are the possible flag values for the link renderer.
// Only a single one of these values will be used; they are not ORed together.
// These are mostly of interest if you are writing a new output format.
const (
	LINK_TYPE_NOT_AUTOLINK = iota
	LINK_TYPE_NORMAL
	LINK_TYPE_EMAIL
)

// These are the possible flag values for the ListItem renderer.
// Multiple flag values may be ORed together.
// These are mostly of interest if you are writing a new output format.
const (
	LIST_TYPE_ORDERED = 1 << iota
	LIST_ITEM_CONTAINS_BLOCK
	LIST_ITEM_BEGINNING_OF_LIST
	LIST_ITEM_END_OF_LIST
)

// These are the possible flag values for the table cell renderer.
// Only a single one of these values will be used; they are not ORed together.
// These are mostly of interest if you are writing a new output format.
const (
	TABLE_ALIGNMENT_LEFT = 1 << iota
	TABLE_ALIGNMENT_RIGHT
	TABLE_ALIGNMENT_CENTER = (TABLE_ALIGNMENT_LEFT | TABLE_ALIGNMENT_RIGHT)
)

// The size of a tab stop.
const (
	TAB_SIZE_DEFAULT = 4
	TAB_SIZE_EIGHT   = 8
)

// These are the tags that are recognized as HTML block tags.
// Any of these can be included in markdown text without special escaping.
var blockTags = map[string]bool{
	"p":          true,
	"dl":         true,
	"h1":         true,
	"h2":         true,
	"h3":         true,
	"h4":         true,
	"h5":         true,
	"h6":         true,
	"ol":         true,
	"ul":         true,
	"del":        true,
	"div":        true,
	"ins":        true,
	"pre":        true,
	"form":       true,
	"math":       true,
	"table":      true,
	"iframe":     true,
	"script":     true,
	"fieldset":   true,
	"noscript":   true,
	"blockquote": true,

	// HTML5
	"video":      true,
	"aside":      true,
	"canvas":     true,
	"figure":     true,
	"footer":     true,
	"header":     true,
	"hgroup":     true,
	"output":     true,
	"article":    true,
	"section":    true,
	"progress":   true,
	"figcaption": true,
}

// Renderer is the rendering interface.
// This is mostly of interest if you are implementing a new rendering format.
//
// When a byte slice is provided, it contains the (rendered) contents of the
// element.
//
// When a callback is provided instead, it will write the contents of the
// respective element directly to the output buffer and return true on success.
// If the callback returns false, the rendering function should reset the
// output buffer as though it had never been called.
//
// Currently Html and Latex implementations are provided
type Renderer interface {
	// block-level callbacks
	BlockCode(out *bytes.Buffer, text []byte, lang string)
	BlockQuote(out *bytes.Buffer, text []byte)
	BlockHtml(out *bytes.Buffer, text []byte)
	Header(out *bytes.Buffer, text func() bool, level int, id string)
	HRule(out *bytes.Buffer)
	List(out *bytes.Buffer, text func() bool, flags int)
	ListItem(out *bytes.Buffer, text []byte, flags int)
	Paragraph(out *bytes.Buffer, text func() bool)
	Table(out *bytes.Buffer, header []byte, body []byte, columnData []int)
	TableRow(out *bytes.Buffer, text []byte)
	TableHeaderCell(out *bytes.Buffer, text []byte, flags int)
	TableCell(out *bytes.Buffer, text []byte, flags int)
	Footnotes(out *bytes.Buffer, text func() bool)
	FootnoteItem(out *bytes.Buffer, name, text []byte, flags int)
	TitleBlock(out *bytes.Buffer, text []byte)

	// Span-level callbacks
	AutoLink(out *bytes.Buffer, link []byte, kind int)
	CodeSpan(out *bytes.Buffer, text []byte)
	DoubleEmphasis(out *bytes.Buffer, text []byte)
	Emphasis(out *bytes.Buffer, text []byte)
	Image(out *bytes.Buffer, link []byte, title []byte, alt []byte)
	LineBreak(out *bytes.Buffer)
	Link(out *bytes.Buffer, link []byte, title []byte, content []byte)
	RawHtmlTag(out *bytes.Buffer, tag []byte)
	TripleEmphasis(out *bytes.Buffer, text []byte)
	StrikeThrough(out *bytes.Buffer, text []byte)
	FootnoteRef(out *bytes.Buffer, ref []byte, id int)

	// Low-level callbacks
	Entity(out *bytes.Buffer, entity []byte)
	NormalText(out *bytes.Buffer, text []byte)

	// Header and footer
	DocumentHeader(out *bytes.Buffer)
	DocumentFooter(out *bytes.Buffer)

	GetFlags() int
}

// Callback functions for inline parsing. One such function is defined
// for each character that triggers a response when parsing inline data.
type inlineParser func(p *parser, out *bytes.Buffer, data []byte, offset int) int

// Parser holds runtime state used by the parser.
// This is constructed by the Markdown function.
type parser struct {
	r              Renderer
	refs           map[string]*reference
	inlineCallback [256]inlineParser
	flags          int
	nesting        int
	maxNesting     int
	insideLink     bool

	// Footnotes need to be ordered as well as available to quickly check for
	// presence. If a ref is also a footnote, it's stored both in refs and here
	// in notes. Slice is nil if footnotes not enabled.
	notes []*reference
}

//
//
// Public interface
//
//

// MarkdownBasic is a convenience function for simple rendering.
// It processes markdown input with no extensions enabled.
func MarkdownBasic(input []byte) []byte {
	// set up the HTML renderer
	htmlFlags := HTML_USE_XHTML
	renderer := HtmlRenderer(htmlFlags, "", "")

	// set up the parser
	extensions := 0

	return Markdown(input, renderer, extensions)
}

// Call Markdown with most useful extensions enabled
// MarkdownCommon is a convenience function for simple rendering.
// It processes markdown input with common extensions enabled, including:
//
// * Smartypants processing with smart fractions and LaTeX dashes
//
// * Intra-word emphasis suppression
//
// * Tables
//
// * Fenced code blocks
//
// * Autolinking
//
// * Strikethrough support
//
// * Strict header parsing
//
// * Custom Header IDs
func MarkdownCommon(input []byte) []byte {
	// set up the HTML renderer
	renderer := HtmlRenderer(commonHtmlFlags, "", "")
	return Markdown(input, renderer, commonExtensions)
}

// Markdown is the main rendering function.
// It parses and renders a block of markdown-encoded text.
// The supplied Renderer is used to format the output, and extensions dictates
// which non-standard extensions are enabled.
//
// To use the supplied Html or LaTeX renderers, see HtmlRenderer and
// LatexRenderer, respectively.
func Markdown(input []byte, renderer Renderer, extensions int) []byte {
	// no point in parsing if we can't render
	if renderer == nil {
		return nil
	}

	// fill in the render structure
	p := new(parser)
	p.r = renderer
	p.flags = extensions
	p.refs = make(map[string]*reference)
	p.maxNesting = 16
	p.insideLink = false

	// register inline parsers
	p.inlineCallback['*'] = emphasis
	p.inlineCallback['_'] = emphasis
	if extensions&EXTENSION_STRIKETHROUGH != 0 {
		p.inlineCallback['~'] = emphasis
	}
	p.inlineCallback['`'] = codeSpan
	p.inlineCallback['\n'] = lineBreak
	p.inlineCallback['['] = link
	p.inlineCallback['<'] = leftAngle
	p.inlineCallback['\\'] = escape
	p.inlineCallback['&'] = entity

	if extensions&EXTENSION_AUTOLINK != 0 {
		p.inlineCallback[':'] = autoLink
	}

	if extensions&EXTENSION_FOOTNOTES != 0 {
		p.notes = make([]*reference, 0)
	}

	first := firstPass(p, input)
	second := secondPass(p, first)
	return second
}

// first pass:
// - extract references
// - expand tabs
// - normalize newlines
// - copy everything else
// - add missing newlines before fenced code blocks
func firstPass(p *parser, input []byte) []byte {
	var out bytes.Buffer
	tabSize := TAB_SIZE_DEFAULT
	if p.flags&EXTENSION_TAB_SIZE_EIGHT != 0 {
		tabSize = TAB_SIZE_EIGHT
	}
	beg, end := 0, 0
	lastLineWasBlank := false
	lastFencedCodeBlockEnd := 0
	for beg < len(input) { // iterate over lines
		if end = isReference(p, input[beg:], tabSize); end > 0 {
			beg += end
		} else { // skip to the next line
			end = beg
			for end < len(input) && input[end] != '\n' && input[end] != '\r' {
				end++
			}

			if p.flags&EXTENSION_FENCED_CODE != 0 {
				// when last line was none blank and a fenced code block comes after
				if beg >= lastFencedCodeBlockEnd {
					if i := p.fencedCode(&out, input[beg:], false); i > 0 {
						if !lastLineWasBlank {
							out.WriteByte('\n') // need to inject additional linebreak
						}
						lastFencedCodeBlockEnd = beg + i
					}
				}
				lastLineWasBlank = end == beg
			}

			// add the line body if present
			if end > beg {
				if end < lastFencedCodeBlockEnd { // Do not expand tabs while inside fenced code blocks.
					out.Write(input[beg:end])
				} else {
					expandTabs(&out, input[beg:end], tabSize)
				}
			}
			out.WriteByte('\n')

			if end < len(input) && input[end] == '\r' {
				end++
			}
			if end < len(input) && input[end] == '\n' {
				end++
			}

			beg = end
		}
	}

	// empty input?
	if out.Len() == 0 {
		out.WriteByte('\n')
	}

	return out.Bytes()
}

// second pass: actual rendering
func secondPass(p *parser, input []byte) []byte {
	var output bytes.Buffer

	p.r.DocumentHeader(&output)
	p.block(&output, input)

	if p.flags&EXTENSION_FOOTNOTES != 0 && len(p.notes) > 0 {
		p.r.Footnotes(&output, func() bool {
			flags := LIST_ITEM_BEGINNING_OF_LIST
			for _, ref := range p.notes {
				var buf bytes.Buffer
				if ref.hasBlock {
					flags |= LIST_ITEM_CONTAINS_BLOCK
					p.block(&buf, ref.title)
				} else {
					p.inline(&buf, ref.title)
				}
				p.r.FootnoteItem(&output, ref.link, buf.Bytes(), flags)
				flags &^= LIST_ITEM_BEGINNING_OF_LIST | LIST_ITEM_CONTAINS_BLOCK
			}

			return true
		})
	}

	p.r.DocumentFooter(&output)

	if p.nesting != 0 {
		panic("Nesting level did not end at zero")
	}

	return output.Bytes()
}

//
// Link references
//
// This section implements support for references that (usually) appear
// as footnotes in a document, and can be referenced anywhere in the document.
// The basic format is:
//
//    [1]: http://www.google.com/ "Google"
//    [2]: http://www.github.com/ "Github"
//
// Anywhere in the document, the reference can be linked by referring to its
// label, i.e., 1 and 2 in this example, as in:
//
//    This library is hosted on [Github][2], a git hosting site.
//
// Actual footnotes as specified in Pandoc and supported by some other Markdown
// libraries such as php-markdown are also taken care of. They look like this:
//
//    This sentence needs a bit of further explanation.[^note]
//
//    [^note]: This is the explanation.
//
// Footnotes should be placed at the end of the document in an ordered list.
// Inline footnotes such as:
//
//    Inline footnotes^[Not supported.] also exist.
//
// are not yet supported.

// References are parsed and stored in this struct.
type reference struct {
	link     []byte
	title    []byte
	noteId   int // 0 if not a footnote ref
	hasBlock bool
}

// Check whether or not data starts with a reference link.
// If so, it is parsed and stored in the list of references
// (in the render struct).
// Returns the number of bytes to skip to move past it,
// or zero if the first line is not a reference.
func isReference(p *parser, data []byte, tabSize int) int {
	// up to 3 optional leading spaces
	if len(data) < 4 {
		return 0
	}
	i := 0
	for i < 3 && data[i] == ' ' {
		i++
	}

	noteId := 0

	// id part: anything but a newline between brackets
	if data[i] != '[' {
		return 0
	}
	i++
	if p.flags&EXTENSION_FOOTNOTES != 0 {
		if data[i] == '^' {
			// we can set it to anything here because the proper noteIds will
			// be assigned later during the second pass. It just has to be != 0
			noteId = 1
			i++
		}
	}
	idOffset := i
	for i < len(data) && data[i] != '\n' && data[i] != '\r' && data[i] != ']' {
		i++
	}
	if i >= len(data) || data[i] != ']' {
		return 0
	}
	idEnd := i

	// spacer: colon (space | tab)* newline? (space | tab)*
	i++
	if i >= len(data) || data[i] != ':' {
		return 0
	}
	i++
	for i < len(data) && (data[i] == ' ' || data[i] == '\t') {
		i++
	}
	if i < len(data) && (data[i] == '\n' || data[i] == '\r') {
		i++
		if i < len(data) && data[i] == '\n' && data[i-1] == '\r' {
			i++
		}
	}
	for i < len(data) && (data[i] == ' ' || data[i] == '\t') {
		i++
	}
	if i >= len(data) {
		return 0
	}

	var (
		linkOffset, linkEnd   int
		titleOffset, titleEnd int
		lineEnd               int
		raw                   []byte
		hasBlock              bool
	)

	if p.flags&EXTENSION_FOOTNOTES != 0 && noteId != 0 {
		linkOffset, linkEnd, raw, hasBlock = scanFootnote(p, data, i, tabSize)
		lineEnd = linkEnd
	} else {
		linkOffset, linkEnd, titleOffset, titleEnd, lineEnd = scanLinkRef(p, data, i)
	}
	if lineEnd == 0 {
		return 0
	}

	// a valid ref has been found

	ref := &reference{
		noteId:   noteId,
		hasBlock: hasBlock,
	}

	if noteId > 0 {
		// reusing the link field for the id since footnotes don't have links
		ref.link = data[idOffset:idEnd]
		// if footnote, it's not really a title, it's the contained text
		ref.title = raw
	} else {
		ref.link = data[linkOffset:linkEnd]
		ref.title = data[titleOffset:titleEnd]
	}

	// id matches are case-insensitive
	id := string(bytes.ToLower(data[idOffset:idEnd]))

	p.refs[id] = ref

	return lineEnd
}

func scanLinkRef(p *parser, data []byte, i int) (linkOffset, linkEnd, titleOffset, titleEnd, lineEnd int) {
	// link: whitespace-free sequence, optionally between angle brackets
	if data[i] == '<' {
		i++
	}
	linkOffset = i
	for i < len(data) && data[i] != ' ' && data[i] != '\t' && data[i] != '\n' && data[i] != '\r' {
		i++
	}
	linkEnd = i
	if data[linkOffset] == '<' && data[linkEnd-1] == '>' {
		linkOffset++
		linkEnd--
	}

	// optional spacer: (space | tab)* (newline | '\'' | '"' | '(' )
	for i < len(data) && (data[i] == ' ' || data[i] == '\t') {
		i++
	}
	if i < len(data) && data[i] != '\n' && data[i] != '\r' && data[i] != '\'' && data[i] != '"' && data[i] != '(' {
		return
	}

	// compute end-of-line
	if i >= len(data) || data[i] == '\r' || data[i] == '\n' {
		lineEnd = i
	}
	if i+1 < len(data) && data[i] == '\r' && data[i+1] == '\n' {
		lineEnd++
	}

	// optional (space|tab)* spacer after a newline
	if lineEnd > 0 {
		i = lineEnd + 1
		for i < len(data) && (data[i] == ' ' || data[i] == '\t') {
			i++
		}
	}

	// optional title: any non-newline sequence enclosed in '"() alone on its line
	if i+1 < len(data) && (data[i] == '\'' || data[i] == '"' || data[i] == '(') {
		i++
		titleOffset = i

		// look for EOL
		for i < len(data) && data[i] != '\n' && data[i] != '\r' {
			i++
		}
		if i+1 < len(data) && data[i] == '\n' && data[i+1] == '\r' {
			titleEnd = i + 1
		} else {
			titleEnd = i
		}

		// step back
		i--
		for i > titleOffset && (data[i] == ' ' || data[i] == '\t') {
			i--
		}
		if i > titleOffset && (data[i] == '\'' || data[i] == '"' || data[i] == ')') {
			lineEnd = titleEnd
			titleEnd = i
		}
	}

	return
}

// The first bit of this logic is the same as (*parser).listItem, but the rest
// is much simpler. This function simply finds the entire block and shifts it
// over by one tab if it is indeed a block (just returns the line if it's not).
// blockEnd is the end of the section in the input buffer, and contents is the
// extracted text that was shifted over one tab. It will need to be rendered at
// the end of the document.
func scanFootnote(p *parser, data []byte, i, indentSize int) (blockStart, blockEnd int, contents []byte, hasBlock bool) {
	if i == 0 || len(data) == 0 {
		return
	}

	// skip leading whitespace on first line
	for i < len(data) && data[i] == ' ' {
		i++
	}

	blockStart = i

	// find the end of the line
	blockEnd = i
	for i < len(data) && data[i-1] != '\n' {
		i++
	}

	// get working buffer
	var raw bytes.Buffer

	// put the first line into the working buffer
	raw.Write(data[blockEnd:i])
	blockEnd = i

	// process the following lines
	containsBlankLine := false

gatherLines:
	for blockEnd < len(data) {
		i++

		// find the end of this line
		for i < len(data) && data[i-1] != '\n' {
			i++
		}

		// if it is an empty line, guess that it is part of this item
		// and move on to the next line
		if p.isEmpty(data[blockEnd:i]) > 0 {
			containsBlankLine = true
			blockEnd = i
			continue
		}

		n := 0
		if n = isIndented(data[blockEnd:i], indentSize); n == 0 {
			// this is the end of the block.
			// we don't want to include this last line in the index.
			break gatherLines
		}

		// if there were blank lines before this one, insert a new one now
		if containsBlankLine {
			raw.WriteByte('\n')
			containsBlankLine = false
		}

		// get rid of that first tab, write to buffer
		raw.Write(data[blockEnd+n : i])
		hasBlock = true

		blockEnd = i
	}

	if data[blockEnd-1] != '\n' {
		raw.WriteByte('\n')
	}

	contents = raw.Bytes()

	return
}

//
//
// Miscellaneous helper functions
//
//

// Test if a character is a punctuation symbol.
// Taken from a private function in regexp in the stdlib.
func ispunct(c byte) bool {
	for _, r := range []byte("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") {
		if c == r {
			return true
		}
	}
	return false
}

// Test if a character is a whitespace character.
func isspace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'
}

// Test if a character is letter.
func isletter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// Test if a character is a letter or a digit.
// TODO: check when this is looking for ASCII alnum and when it should use unicode
func isalnum(c byte) bool {
	return (c >= '0' && c <= '9') || isletter(c)
}

// Replace tab characters with spaces, aligning to the next TAB_SIZE column.
// always ends output with a newline
func expandTabs(out *bytes.Buffer, line []byte, tabSize int) {
	// first, check for common cases: no tabs, or only tabs at beginning of line
	i, prefix := 0, 0
	slowcase := false
	for i = 0; i < len(line); i++ {
		if line[i] == '\t' {
			if prefix == i {
				prefix++
			} else {
				slowcase = true
				break
			}
		}
	}

	// no need to decode runes if all tabs are at the beginning of the line
	if !slowcase {
		for i = 0; i < prefix*tabSize; i++ {
			out.WriteByte(' ')
		}
		out.Write(line[prefix:])
		return
	}

	// the slow case: we need to count runes to figure out how
	// many spaces to insert for each tab
	column := 0
	i = 0
	for i < len(line) {
		start := i
		for i < len(line) && line[i] != '\t' {
			_, size := utf8.DecodeRune(line[i:])
			i += size
			column++
		}

		if i > start {
			out.Write(line[start:i])
		}

		if i >= len(line) {
			break
		}

		for {
			out.WriteByte(' ')
			column++
			if column%tabSize == 0 {
				break
			}
		}

		i++
	}
}

// Find if a line counts as indented or not.
// Returns number of characters the indent is (0 = not indented).
func isIndented(data []byte, indentSize int) int {
	if len(data) == 0 {
		return 0
	}
	if data[0] == '\t' {
		return 1
	}
	if len(data) < indentSize {
		return 0
	}
	for i := 0; i < indentSize; i++ {
		if data[i] != ' ' {
			return 0
		}
	}
	return indentSize
}

// Create a url-safe slug for fragments
func slugify(in []byte) []byte {
	if len(in) == 0 {
		return in
	}
	out := make([]byte, 0, len(in))
	sym := false

	for _, ch := range in {
		if isalnum(ch) {
			sym = false
			out = append(out, ch)
		} else if sym {
			continue
		} else {
			out = append(out, '-')
			sym = true
		}
	}
	var a, b int
	var ch byte
	for a, ch = range out {
		if ch != '-' {
			break
		}
	}
	for b = len(out) - 1; b > 0; b-- {
		if out[b] != '-' {
			break
		}
	}
	return out[a : b+1]
}
