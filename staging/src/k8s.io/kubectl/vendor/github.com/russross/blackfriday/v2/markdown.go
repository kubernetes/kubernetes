// Blackfriday Markdown Processor
// Available at http://github.com/russross/blackfriday
//
// Copyright © 2011 Russ Ross <russ@russross.com>.
// Distributed under the Simplified BSD License.
// See README.md for details.

package blackfriday

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

//
// Markdown parsing and processing
//

// Version string of the package. Appears in the rendered document when
// CompletePage flag is on.
const Version = "2.0"

// Extensions is a bitwise or'ed collection of enabled Blackfriday's
// extensions.
type Extensions int

// These are the supported markdown parsing extensions.
// OR these values together to select multiple extensions.
const (
	NoExtensions           Extensions = 0
	NoIntraEmphasis        Extensions = 1 << iota // Ignore emphasis markers inside words
	Tables                                        // Render tables
	FencedCode                                    // Render fenced code blocks
	Autolink                                      // Detect embedded URLs that are not explicitly marked
	Strikethrough                                 // Strikethrough text using ~~test~~
	LaxHTMLBlocks                                 // Loosen up HTML block parsing rules
	SpaceHeadings                                 // Be strict about prefix heading rules
	HardLineBreak                                 // Translate newlines into line breaks
	TabSizeEight                                  // Expand tabs to eight spaces instead of four
	Footnotes                                     // Pandoc-style footnotes
	NoEmptyLineBeforeBlock                        // No need to insert an empty line to start a (code, quote, ordered list, unordered list) block
	HeadingIDs                                    // specify heading IDs  with {#id}
	Titleblock                                    // Titleblock ala pandoc
	AutoHeadingIDs                                // Create the heading ID from the text
	BackslashLineBreak                            // Translate trailing backslashes into line breaks
	DefinitionLists                               // Render definition lists

	CommonHTMLFlags HTMLFlags = UseXHTML | Smartypants |
		SmartypantsFractions | SmartypantsDashes | SmartypantsLatexDashes

	CommonExtensions Extensions = NoIntraEmphasis | Tables | FencedCode |
		Autolink | Strikethrough | SpaceHeadings | HeadingIDs |
		BackslashLineBreak | DefinitionLists
)

// ListType contains bitwise or'ed flags for list and list item objects.
type ListType int

// These are the possible flag values for the ListItem renderer.
// Multiple flag values may be ORed together.
// These are mostly of interest if you are writing a new output format.
const (
	ListTypeOrdered ListType = 1 << iota
	ListTypeDefinition
	ListTypeTerm

	ListItemContainsBlock
	ListItemBeginningOfList // TODO: figure out if this is of any use now
	ListItemEndOfList
)

// CellAlignFlags holds a type of alignment in a table cell.
type CellAlignFlags int

// These are the possible flag values for the table cell renderer.
// Only a single one of these values will be used; they are not ORed together.
// These are mostly of interest if you are writing a new output format.
const (
	TableAlignmentLeft CellAlignFlags = 1 << iota
	TableAlignmentRight
	TableAlignmentCenter = (TableAlignmentLeft | TableAlignmentRight)
)

// The size of a tab stop.
const (
	TabSizeDefault = 4
	TabSizeDouble  = 8
)

// blockTags is a set of tags that are recognized as HTML block tags.
// Any of these can be included in markdown text without special escaping.
var blockTags = map[string]struct{}{
	"blockquote": {},
	"del":        {},
	"div":        {},
	"dl":         {},
	"fieldset":   {},
	"form":       {},
	"h1":         {},
	"h2":         {},
	"h3":         {},
	"h4":         {},
	"h5":         {},
	"h6":         {},
	"iframe":     {},
	"ins":        {},
	"math":       {},
	"noscript":   {},
	"ol":         {},
	"pre":        {},
	"p":          {},
	"script":     {},
	"style":      {},
	"table":      {},
	"ul":         {},

	// HTML5
	"address":    {},
	"article":    {},
	"aside":      {},
	"canvas":     {},
	"figcaption": {},
	"figure":     {},
	"footer":     {},
	"header":     {},
	"hgroup":     {},
	"main":       {},
	"nav":        {},
	"output":     {},
	"progress":   {},
	"section":    {},
	"video":      {},
}

// Renderer is the rendering interface. This is mostly of interest if you are
// implementing a new rendering format.
//
// Only an HTML implementation is provided in this repository, see the README
// for external implementations.
type Renderer interface {
	// RenderNode is the main rendering method. It will be called once for
	// every leaf node and twice for every non-leaf node (first with
	// entering=true, then with entering=false). The method should write its
	// rendition of the node to the supplied writer w.
	RenderNode(w io.Writer, node *Node, entering bool) WalkStatus

	// RenderHeader is a method that allows the renderer to produce some
	// content preceding the main body of the output document. The header is
	// understood in the broad sense here. For example, the default HTML
	// renderer will write not only the HTML document preamble, but also the
	// table of contents if it was requested.
	//
	// The method will be passed an entire document tree, in case a particular
	// implementation needs to inspect it to produce output.
	//
	// The output should be written to the supplied writer w. If your
	// implementation has no header to write, supply an empty implementation.
	RenderHeader(w io.Writer, ast *Node)

	// RenderFooter is a symmetric counterpart of RenderHeader.
	RenderFooter(w io.Writer, ast *Node)
}

// Callback functions for inline parsing. One such function is defined
// for each character that triggers a response when parsing inline data.
type inlineParser func(p *Markdown, data []byte, offset int) (int, *Node)

// Markdown is a type that holds extensions and the runtime state used by
// Parse, and the renderer. You can not use it directly, construct it with New.
type Markdown struct {
	renderer          Renderer
	referenceOverride ReferenceOverrideFunc
	refs              map[string]*reference
	inlineCallback    [256]inlineParser
	extensions        Extensions
	nesting           int
	maxNesting        int
	insideLink        bool

	// Footnotes need to be ordered as well as available to quickly check for
	// presence. If a ref is also a footnote, it's stored both in refs and here
	// in notes. Slice is nil if footnotes not enabled.
	notes []*reference

	doc                  *Node
	tip                  *Node // = doc
	oldTip               *Node
	lastMatchedContainer *Node // = doc
	allClosed            bool
}

func (p *Markdown) getRef(refid string) (ref *reference, found bool) {
	if p.referenceOverride != nil {
		r, overridden := p.referenceOverride(refid)
		if overridden {
			if r == nil {
				return nil, false
			}
			return &reference{
				link:     []byte(r.Link),
				title:    []byte(r.Title),
				noteID:   0,
				hasBlock: false,
				text:     []byte(r.Text)}, true
		}
	}
	// refs are case insensitive
	ref, found = p.refs[strings.ToLower(refid)]
	return ref, found
}

func (p *Markdown) finalize(block *Node) {
	above := block.Parent
	block.open = false
	p.tip = above
}

func (p *Markdown) addChild(node NodeType, offset uint32) *Node {
	return p.addExistingChild(NewNode(node), offset)
}

func (p *Markdown) addExistingChild(node *Node, offset uint32) *Node {
	for !p.tip.canContain(node.Type) {
		p.finalize(p.tip)
	}
	p.tip.AppendChild(node)
	p.tip = node
	return node
}

func (p *Markdown) closeUnmatchedBlocks() {
	if !p.allClosed {
		for p.oldTip != p.lastMatchedContainer {
			parent := p.oldTip.Parent
			p.finalize(p.oldTip)
			p.oldTip = parent
		}
		p.allClosed = true
	}
}

//
//
// Public interface
//
//

// Reference represents the details of a link.
// See the documentation in Options for more details on use-case.
type Reference struct {
	// Link is usually the URL the reference points to.
	Link string
	// Title is the alternate text describing the link in more detail.
	Title string
	// Text is the optional text to override the ref with if the syntax used was
	// [refid][]
	Text string
}

// ReferenceOverrideFunc is expected to be called with a reference string and
// return either a valid Reference type that the reference string maps to or
// nil. If overridden is false, the default reference logic will be executed.
// See the documentation in Options for more details on use-case.
type ReferenceOverrideFunc func(reference string) (ref *Reference, overridden bool)

// New constructs a Markdown processor. You can use the same With* functions as
// for Run() to customize parser's behavior and the renderer.
func New(opts ...Option) *Markdown {
	var p Markdown
	for _, opt := range opts {
		opt(&p)
	}
	p.refs = make(map[string]*reference)
	p.maxNesting = 16
	p.insideLink = false
	docNode := NewNode(Document)
	p.doc = docNode
	p.tip = docNode
	p.oldTip = docNode
	p.lastMatchedContainer = docNode
	p.allClosed = true
	// register inline parsers
	p.inlineCallback[' '] = maybeLineBreak
	p.inlineCallback['*'] = emphasis
	p.inlineCallback['_'] = emphasis
	if p.extensions&Strikethrough != 0 {
		p.inlineCallback['~'] = emphasis
	}
	p.inlineCallback['`'] = codeSpan
	p.inlineCallback['\n'] = lineBreak
	p.inlineCallback['['] = link
	p.inlineCallback['<'] = leftAngle
	p.inlineCallback['\\'] = escape
	p.inlineCallback['&'] = entity
	p.inlineCallback['!'] = maybeImage
	p.inlineCallback['^'] = maybeInlineFootnote
	if p.extensions&Autolink != 0 {
		p.inlineCallback['h'] = maybeAutoLink
		p.inlineCallback['m'] = maybeAutoLink
		p.inlineCallback['f'] = maybeAutoLink
		p.inlineCallback['H'] = maybeAutoLink
		p.inlineCallback['M'] = maybeAutoLink
		p.inlineCallback['F'] = maybeAutoLink
	}
	if p.extensions&Footnotes != 0 {
		p.notes = make([]*reference, 0)
	}
	return &p
}

// Option customizes the Markdown processor's default behavior.
type Option func(*Markdown)

// WithRenderer allows you to override the default renderer.
func WithRenderer(r Renderer) Option {
	return func(p *Markdown) {
		p.renderer = r
	}
}

// WithExtensions allows you to pick some of the many extensions provided by
// Blackfriday. You can bitwise OR them.
func WithExtensions(e Extensions) Option {
	return func(p *Markdown) {
		p.extensions = e
	}
}

// WithNoExtensions turns off all extensions and custom behavior.
func WithNoExtensions() Option {
	return func(p *Markdown) {
		p.extensions = NoExtensions
		p.renderer = NewHTMLRenderer(HTMLRendererParameters{
			Flags: HTMLFlagsNone,
		})
	}
}

// WithRefOverride sets an optional function callback that is called every
// time a reference is resolved.
//
// In Markdown, the link reference syntax can be made to resolve a link to
// a reference instead of an inline URL, in one of the following ways:
//
//  * [link text][refid]
//  * [refid][]
//
// Usually, the refid is defined at the bottom of the Markdown document. If
// this override function is provided, the refid is passed to the override
// function first, before consulting the defined refids at the bottom. If
// the override function indicates an override did not occur, the refids at
// the bottom will be used to fill in the link details.
func WithRefOverride(o ReferenceOverrideFunc) Option {
	return func(p *Markdown) {
		p.referenceOverride = o
	}
}

// Run is the main entry point to Blackfriday. It parses and renders a
// block of markdown-encoded text.
//
// The simplest invocation of Run takes one argument, input:
//     output := Run(input)
// This will parse the input with CommonExtensions enabled and render it with
// the default HTMLRenderer (with CommonHTMLFlags).
//
// Variadic arguments opts can customize the default behavior. Since Markdown
// type does not contain exported fields, you can not use it directly. Instead,
// use the With* functions. For example, this will call the most basic
// functionality, with no extensions:
//     output := Run(input, WithNoExtensions())
//
// You can use any number of With* arguments, even contradicting ones. They
// will be applied in order of appearance and the latter will override the
// former:
//     output := Run(input, WithNoExtensions(), WithExtensions(exts),
//         WithRenderer(yourRenderer))
func Run(input []byte, opts ...Option) []byte {
	r := NewHTMLRenderer(HTMLRendererParameters{
		Flags: CommonHTMLFlags,
	})
	optList := []Option{WithRenderer(r), WithExtensions(CommonExtensions)}
	optList = append(optList, opts...)
	parser := New(optList...)
	ast := parser.Parse(input)
	var buf bytes.Buffer
	parser.renderer.RenderHeader(&buf, ast)
	ast.Walk(func(node *Node, entering bool) WalkStatus {
		return parser.renderer.RenderNode(&buf, node, entering)
	})
	parser.renderer.RenderFooter(&buf, ast)
	return buf.Bytes()
}

// Parse is an entry point to the parsing part of Blackfriday. It takes an
// input markdown document and produces a syntax tree for its contents. This
// tree can then be rendered with a default or custom renderer, or
// analyzed/transformed by the caller to whatever non-standard needs they have.
// The return value is the root node of the syntax tree.
func (p *Markdown) Parse(input []byte) *Node {
	p.block(input)
	// Walk the tree and finish up some of unfinished blocks
	for p.tip != nil {
		p.finalize(p.tip)
	}
	// Walk the tree again and process inline markdown in each block
	p.doc.Walk(func(node *Node, entering bool) WalkStatus {
		if node.Type == Paragraph || node.Type == Heading || node.Type == TableCell {
			p.inline(node, node.content)
			node.content = nil
		}
		return GoToNext
	})
	p.parseRefsToAST()
	return p.doc
}

func (p *Markdown) parseRefsToAST() {
	if p.extensions&Footnotes == 0 || len(p.notes) == 0 {
		return
	}
	p.tip = p.doc
	block := p.addBlock(List, nil)
	block.IsFootnotesList = true
	block.ListFlags = ListTypeOrdered
	flags := ListItemBeginningOfList
	// Note: this loop is intentionally explicit, not range-form. This is
	// because the body of the loop will append nested footnotes to p.notes and
	// we need to process those late additions. Range form would only walk over
	// the fixed initial set.
	for i := 0; i < len(p.notes); i++ {
		ref := p.notes[i]
		p.addExistingChild(ref.footnote, 0)
		block := ref.footnote
		block.ListFlags = flags | ListTypeOrdered
		block.RefLink = ref.link
		if ref.hasBlock {
			flags |= ListItemContainsBlock
			p.block(ref.title)
		} else {
			p.inline(block, ref.title)
		}
		flags &^= ListItemBeginningOfList | ListItemContainsBlock
	}
	above := block.Parent
	finalizeList(block)
	p.tip = above
	block.Walk(func(node *Node, entering bool) WalkStatus {
		if node.Type == Paragraph || node.Type == Heading {
			p.inline(node, node.content)
			node.content = nil
		}
		return GoToNext
	})
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
// Finally, there are inline footnotes such as:
//
//    Inline footnotes^[Also supported.] provide a quick inline explanation,
//    but are rendered at the bottom of the document.
//

// reference holds all information necessary for a reference-style links or
// footnotes.
//
// Consider this markdown with reference-style links:
//
//     [link][ref]
//
//     [ref]: /url/ "tooltip title"
//
// It will be ultimately converted to this HTML:
//
//     <p><a href=\"/url/\" title=\"title\">link</a></p>
//
// And a reference structure will be populated as follows:
//
//     p.refs["ref"] = &reference{
//         link: "/url/",
//         title: "tooltip title",
//     }
//
// Alternatively, reference can contain information about a footnote. Consider
// this markdown:
//
//     Text needing a footnote.[^a]
//
//     [^a]: This is the note
//
// A reference structure will be populated as follows:
//
//     p.refs["a"] = &reference{
//         link: "a",
//         title: "This is the note",
//         noteID: <some positive int>,
//     }
//
// TODO: As you can see, it begs for splitting into two dedicated structures
// for refs and for footnotes.
type reference struct {
	link     []byte
	title    []byte
	noteID   int // 0 if not a footnote ref
	hasBlock bool
	footnote *Node // a link to the Item node within a list of footnotes

	text []byte // only gets populated by refOverride feature with Reference.Text
}

func (r *reference) String() string {
	return fmt.Sprintf("{link: %q, title: %q, text: %q, noteID: %d, hasBlock: %v}",
		r.link, r.title, r.text, r.noteID, r.hasBlock)
}

// Check whether or not data starts with a reference link.
// If so, it is parsed and stored in the list of references
// (in the render struct).
// Returns the number of bytes to skip to move past it,
// or zero if the first line is not a reference.
func isReference(p *Markdown, data []byte, tabSize int) int {
	// up to 3 optional leading spaces
	if len(data) < 4 {
		return 0
	}
	i := 0
	for i < 3 && data[i] == ' ' {
		i++
	}

	noteID := 0

	// id part: anything but a newline between brackets
	if data[i] != '[' {
		return 0
	}
	i++
	if p.extensions&Footnotes != 0 {
		if i < len(data) && data[i] == '^' {
			// we can set it to anything here because the proper noteIds will
			// be assigned later during the second pass. It just has to be != 0
			noteID = 1
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
	// footnotes can have empty ID, like this: [^], but a reference can not be
	// empty like this: []. Break early if it's not a footnote and there's no ID
	if noteID == 0 && idOffset == idEnd {
		return 0
	}
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

	if p.extensions&Footnotes != 0 && noteID != 0 {
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
		noteID:   noteID,
		hasBlock: hasBlock,
	}

	if noteID > 0 {
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

func scanLinkRef(p *Markdown, data []byte, i int) (linkOffset, linkEnd, titleOffset, titleEnd, lineEnd int) {
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

// The first bit of this logic is the same as Parser.listItem, but the rest
// is much simpler. This function simply finds the entire block and shifts it
// over by one tab if it is indeed a block (just returns the line if it's not).
// blockEnd is the end of the section in the input buffer, and contents is the
// extracted text that was shifted over one tab. It will need to be rendered at
// the end of the document.
func scanFootnote(p *Markdown, data []byte, i, indentSize int) (blockStart, blockEnd int, contents []byte, hasBlock bool) {
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
	return ishorizontalspace(c) || isverticalspace(c)
}

// Test if a character is a horizontal whitespace character.
func ishorizontalspace(c byte) bool {
	return c == ' ' || c == '\t'
}

// Test if a character is a vertical character.
func isverticalspace(c byte) bool {
	return c == '\n' || c == '\r' || c == '\f' || c == '\v'
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
