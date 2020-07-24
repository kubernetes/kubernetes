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
// HTML rendering backend
//
//

package blackfriday

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strings"
)

// HTMLFlags control optional behavior of HTML renderer.
type HTMLFlags int

// HTML renderer configuration options.
const (
	HTMLFlagsNone           HTMLFlags = 0
	SkipHTML                HTMLFlags = 1 << iota // Skip preformatted HTML blocks
	SkipImages                                    // Skip embedded images
	SkipLinks                                     // Skip all links
	Safelink                                      // Only link to trusted protocols
	NofollowLinks                                 // Only link with rel="nofollow"
	NoreferrerLinks                               // Only link with rel="noreferrer"
	NoopenerLinks                                 // Only link with rel="noopener"
	HrefTargetBlank                               // Add a blank target
	CompletePage                                  // Generate a complete HTML page
	UseXHTML                                      // Generate XHTML output instead of HTML
	FootnoteReturnLinks                           // Generate a link at the end of a footnote to return to the source
	Smartypants                                   // Enable smart punctuation substitutions
	SmartypantsFractions                          // Enable smart fractions (with Smartypants)
	SmartypantsDashes                             // Enable smart dashes (with Smartypants)
	SmartypantsLatexDashes                        // Enable LaTeX-style dashes (with Smartypants)
	SmartypantsAngledQuotes                       // Enable angled double quotes (with Smartypants) for double quotes rendering
	SmartypantsQuotesNBSP                         // Enable « French guillemets » (with Smartypants)
	TOC                                           // Generate a table of contents
)

var (
	htmlTagRe = regexp.MustCompile("(?i)^" + htmlTag)
)

const (
	htmlTag = "(?:" + openTag + "|" + closeTag + "|" + htmlComment + "|" +
		processingInstruction + "|" + declaration + "|" + cdata + ")"
	closeTag              = "</" + tagName + "\\s*[>]"
	openTag               = "<" + tagName + attribute + "*" + "\\s*/?>"
	attribute             = "(?:" + "\\s+" + attributeName + attributeValueSpec + "?)"
	attributeValue        = "(?:" + unquotedValue + "|" + singleQuotedValue + "|" + doubleQuotedValue + ")"
	attributeValueSpec    = "(?:" + "\\s*=" + "\\s*" + attributeValue + ")"
	attributeName         = "[a-zA-Z_:][a-zA-Z0-9:._-]*"
	cdata                 = "<!\\[CDATA\\[[\\s\\S]*?\\]\\]>"
	declaration           = "<![A-Z]+" + "\\s+[^>]*>"
	doubleQuotedValue     = "\"[^\"]*\""
	htmlComment           = "<!---->|<!--(?:-?[^>-])(?:-?[^-])*-->"
	processingInstruction = "[<][?].*?[?][>]"
	singleQuotedValue     = "'[^']*'"
	tagName               = "[A-Za-z][A-Za-z0-9-]*"
	unquotedValue         = "[^\"'=<>`\\x00-\\x20]+"
)

// HTMLRendererParameters is a collection of supplementary parameters tweaking
// the behavior of various parts of HTML renderer.
type HTMLRendererParameters struct {
	// Prepend this text to each relative URL.
	AbsolutePrefix string
	// Add this text to each footnote anchor, to ensure uniqueness.
	FootnoteAnchorPrefix string
	// Show this text inside the <a> tag for a footnote return link, if the
	// HTML_FOOTNOTE_RETURN_LINKS flag is enabled. If blank, the string
	// <sup>[return]</sup> is used.
	FootnoteReturnLinkContents string
	// If set, add this text to the front of each Heading ID, to ensure
	// uniqueness.
	HeadingIDPrefix string
	// If set, add this text to the back of each Heading ID, to ensure uniqueness.
	HeadingIDSuffix string
	// Increase heading levels: if the offset is 1, <h1> becomes <h2> etc.
	// Negative offset is also valid.
	// Resulting levels are clipped between 1 and 6.
	HeadingLevelOffset int

	Title string // Document title (used if CompletePage is set)
	CSS   string // Optional CSS file URL (used if CompletePage is set)
	Icon  string // Optional icon file URL (used if CompletePage is set)

	Flags HTMLFlags // Flags allow customizing this renderer's behavior
}

// HTMLRenderer is a type that implements the Renderer interface for HTML output.
//
// Do not create this directly, instead use the NewHTMLRenderer function.
type HTMLRenderer struct {
	HTMLRendererParameters

	closeTag string // how to end singleton tags: either " />" or ">"

	// Track heading IDs to prevent ID collision in a single generation.
	headingIDs map[string]int

	lastOutputLen int
	disableTags   int

	sr *SPRenderer
}

const (
	xhtmlClose = " />"
	htmlClose  = ">"
)

// NewHTMLRenderer creates and configures an HTMLRenderer object, which
// satisfies the Renderer interface.
func NewHTMLRenderer(params HTMLRendererParameters) *HTMLRenderer {
	// configure the rendering engine
	closeTag := htmlClose
	if params.Flags&UseXHTML != 0 {
		closeTag = xhtmlClose
	}

	if params.FootnoteReturnLinkContents == "" {
		params.FootnoteReturnLinkContents = `<sup>[return]</sup>`
	}

	return &HTMLRenderer{
		HTMLRendererParameters: params,

		closeTag:   closeTag,
		headingIDs: make(map[string]int),

		sr: NewSmartypantsRenderer(params.Flags),
	}
}

func isHTMLTag(tag []byte, tagname string) bool {
	found, _ := findHTMLTagPos(tag, tagname)
	return found
}

// Look for a character, but ignore it when it's in any kind of quotes, it
// might be JavaScript
func skipUntilCharIgnoreQuotes(html []byte, start int, char byte) int {
	inSingleQuote := false
	inDoubleQuote := false
	inGraveQuote := false
	i := start
	for i < len(html) {
		switch {
		case html[i] == char && !inSingleQuote && !inDoubleQuote && !inGraveQuote:
			return i
		case html[i] == '\'':
			inSingleQuote = !inSingleQuote
		case html[i] == '"':
			inDoubleQuote = !inDoubleQuote
		case html[i] == '`':
			inGraveQuote = !inGraveQuote
		}
		i++
	}
	return start
}

func findHTMLTagPos(tag []byte, tagname string) (bool, int) {
	i := 0
	if i < len(tag) && tag[0] != '<' {
		return false, -1
	}
	i++
	i = skipSpace(tag, i)

	if i < len(tag) && tag[i] == '/' {
		i++
	}

	i = skipSpace(tag, i)
	j := 0
	for ; i < len(tag); i, j = i+1, j+1 {
		if j >= len(tagname) {
			break
		}

		if strings.ToLower(string(tag[i]))[0] != tagname[j] {
			return false, -1
		}
	}

	if i == len(tag) {
		return false, -1
	}

	rightAngle := skipUntilCharIgnoreQuotes(tag, i, '>')
	if rightAngle >= i {
		return true, rightAngle
	}

	return false, -1
}

func skipSpace(tag []byte, i int) int {
	for i < len(tag) && isspace(tag[i]) {
		i++
	}
	return i
}

func isRelativeLink(link []byte) (yes bool) {
	// a tag begin with '#'
	if link[0] == '#' {
		return true
	}

	// link begin with '/' but not '//', the second maybe a protocol relative link
	if len(link) >= 2 && link[0] == '/' && link[1] != '/' {
		return true
	}

	// only the root '/'
	if len(link) == 1 && link[0] == '/' {
		return true
	}

	// current directory : begin with "./"
	if bytes.HasPrefix(link, []byte("./")) {
		return true
	}

	// parent directory : begin with "../"
	if bytes.HasPrefix(link, []byte("../")) {
		return true
	}

	return false
}

func (r *HTMLRenderer) ensureUniqueHeadingID(id string) string {
	for count, found := r.headingIDs[id]; found; count, found = r.headingIDs[id] {
		tmp := fmt.Sprintf("%s-%d", id, count+1)

		if _, tmpFound := r.headingIDs[tmp]; !tmpFound {
			r.headingIDs[id] = count + 1
			id = tmp
		} else {
			id = id + "-1"
		}
	}

	if _, found := r.headingIDs[id]; !found {
		r.headingIDs[id] = 0
	}

	return id
}

func (r *HTMLRenderer) addAbsPrefix(link []byte) []byte {
	if r.AbsolutePrefix != "" && isRelativeLink(link) && link[0] != '.' {
		newDest := r.AbsolutePrefix
		if link[0] != '/' {
			newDest += "/"
		}
		newDest += string(link)
		return []byte(newDest)
	}
	return link
}

func appendLinkAttrs(attrs []string, flags HTMLFlags, link []byte) []string {
	if isRelativeLink(link) {
		return attrs
	}
	val := []string{}
	if flags&NofollowLinks != 0 {
		val = append(val, "nofollow")
	}
	if flags&NoreferrerLinks != 0 {
		val = append(val, "noreferrer")
	}
	if flags&NoopenerLinks != 0 {
		val = append(val, "noopener")
	}
	if flags&HrefTargetBlank != 0 {
		attrs = append(attrs, "target=\"_blank\"")
	}
	if len(val) == 0 {
		return attrs
	}
	attr := fmt.Sprintf("rel=%q", strings.Join(val, " "))
	return append(attrs, attr)
}

func isMailto(link []byte) bool {
	return bytes.HasPrefix(link, []byte("mailto:"))
}

func needSkipLink(flags HTMLFlags, dest []byte) bool {
	if flags&SkipLinks != 0 {
		return true
	}
	return flags&Safelink != 0 && !isSafeLink(dest) && !isMailto(dest)
}

func isSmartypantable(node *Node) bool {
	pt := node.Parent.Type
	return pt != Link && pt != CodeBlock && pt != Code
}

func appendLanguageAttr(attrs []string, info []byte) []string {
	if len(info) == 0 {
		return attrs
	}
	endOfLang := bytes.IndexAny(info, "\t ")
	if endOfLang < 0 {
		endOfLang = len(info)
	}
	return append(attrs, fmt.Sprintf("class=\"language-%s\"", info[:endOfLang]))
}

func (r *HTMLRenderer) tag(w io.Writer, name []byte, attrs []string) {
	w.Write(name)
	if len(attrs) > 0 {
		w.Write(spaceBytes)
		w.Write([]byte(strings.Join(attrs, " ")))
	}
	w.Write(gtBytes)
	r.lastOutputLen = 1
}

func footnoteRef(prefix string, node *Node) []byte {
	urlFrag := prefix + string(slugify(node.Destination))
	anchor := fmt.Sprintf(`<a href="#fn:%s">%d</a>`, urlFrag, node.NoteID)
	return []byte(fmt.Sprintf(`<sup class="footnote-ref" id="fnref:%s">%s</sup>`, urlFrag, anchor))
}

func footnoteItem(prefix string, slug []byte) []byte {
	return []byte(fmt.Sprintf(`<li id="fn:%s%s">`, prefix, slug))
}

func footnoteReturnLink(prefix, returnLink string, slug []byte) []byte {
	const format = ` <a class="footnote-return" href="#fnref:%s%s">%s</a>`
	return []byte(fmt.Sprintf(format, prefix, slug, returnLink))
}

func itemOpenCR(node *Node) bool {
	if node.Prev == nil {
		return false
	}
	ld := node.Parent.ListData
	return !ld.Tight && ld.ListFlags&ListTypeDefinition == 0
}

func skipParagraphTags(node *Node) bool {
	grandparent := node.Parent.Parent
	if grandparent == nil || grandparent.Type != List {
		return false
	}
	tightOrTerm := grandparent.Tight || node.Parent.ListFlags&ListTypeTerm != 0
	return grandparent.Type == List && tightOrTerm
}

func cellAlignment(align CellAlignFlags) string {
	switch align {
	case TableAlignmentLeft:
		return "left"
	case TableAlignmentRight:
		return "right"
	case TableAlignmentCenter:
		return "center"
	default:
		return ""
	}
}

func (r *HTMLRenderer) out(w io.Writer, text []byte) {
	if r.disableTags > 0 {
		w.Write(htmlTagRe.ReplaceAll(text, []byte{}))
	} else {
		w.Write(text)
	}
	r.lastOutputLen = len(text)
}

func (r *HTMLRenderer) cr(w io.Writer) {
	if r.lastOutputLen > 0 {
		r.out(w, nlBytes)
	}
}

var (
	nlBytes    = []byte{'\n'}
	gtBytes    = []byte{'>'}
	spaceBytes = []byte{' '}
)

var (
	brTag              = []byte("<br>")
	brXHTMLTag         = []byte("<br />")
	emTag              = []byte("<em>")
	emCloseTag         = []byte("</em>")
	strongTag          = []byte("<strong>")
	strongCloseTag     = []byte("</strong>")
	delTag             = []byte("<del>")
	delCloseTag        = []byte("</del>")
	ttTag              = []byte("<tt>")
	ttCloseTag         = []byte("</tt>")
	aTag               = []byte("<a")
	aCloseTag          = []byte("</a>")
	preTag             = []byte("<pre>")
	preCloseTag        = []byte("</pre>")
	codeTag            = []byte("<code>")
	codeCloseTag       = []byte("</code>")
	pTag               = []byte("<p>")
	pCloseTag          = []byte("</p>")
	blockquoteTag      = []byte("<blockquote>")
	blockquoteCloseTag = []byte("</blockquote>")
	hrTag              = []byte("<hr>")
	hrXHTMLTag         = []byte("<hr />")
	ulTag              = []byte("<ul>")
	ulCloseTag         = []byte("</ul>")
	olTag              = []byte("<ol>")
	olCloseTag         = []byte("</ol>")
	dlTag              = []byte("<dl>")
	dlCloseTag         = []byte("</dl>")
	liTag              = []byte("<li>")
	liCloseTag         = []byte("</li>")
	ddTag              = []byte("<dd>")
	ddCloseTag         = []byte("</dd>")
	dtTag              = []byte("<dt>")
	dtCloseTag         = []byte("</dt>")
	tableTag           = []byte("<table>")
	tableCloseTag      = []byte("</table>")
	tdTag              = []byte("<td")
	tdCloseTag         = []byte("</td>")
	thTag              = []byte("<th")
	thCloseTag         = []byte("</th>")
	theadTag           = []byte("<thead>")
	theadCloseTag      = []byte("</thead>")
	tbodyTag           = []byte("<tbody>")
	tbodyCloseTag      = []byte("</tbody>")
	trTag              = []byte("<tr>")
	trCloseTag         = []byte("</tr>")
	h1Tag              = []byte("<h1")
	h1CloseTag         = []byte("</h1>")
	h2Tag              = []byte("<h2")
	h2CloseTag         = []byte("</h2>")
	h3Tag              = []byte("<h3")
	h3CloseTag         = []byte("</h3>")
	h4Tag              = []byte("<h4")
	h4CloseTag         = []byte("</h4>")
	h5Tag              = []byte("<h5")
	h5CloseTag         = []byte("</h5>")
	h6Tag              = []byte("<h6")
	h6CloseTag         = []byte("</h6>")

	footnotesDivBytes      = []byte("\n<div class=\"footnotes\">\n\n")
	footnotesCloseDivBytes = []byte("\n</div>\n")
)

func headingTagsFromLevel(level int) ([]byte, []byte) {
	if level <= 1 {
		return h1Tag, h1CloseTag
	}
	switch level {
	case 2:
		return h2Tag, h2CloseTag
	case 3:
		return h3Tag, h3CloseTag
	case 4:
		return h4Tag, h4CloseTag
	case 5:
		return h5Tag, h5CloseTag
	}
	return h6Tag, h6CloseTag
}

func (r *HTMLRenderer) outHRTag(w io.Writer) {
	if r.Flags&UseXHTML == 0 {
		r.out(w, hrTag)
	} else {
		r.out(w, hrXHTMLTag)
	}
}

// RenderNode is a default renderer of a single node of a syntax tree. For
// block nodes it will be called twice: first time with entering=true, second
// time with entering=false, so that it could know when it's working on an open
// tag and when on close. It writes the result to w.
//
// The return value is a way to tell the calling walker to adjust its walk
// pattern: e.g. it can terminate the traversal by returning Terminate. Or it
// can ask the walker to skip a subtree of this node by returning SkipChildren.
// The typical behavior is to return GoToNext, which asks for the usual
// traversal to the next node.
func (r *HTMLRenderer) RenderNode(w io.Writer, node *Node, entering bool) WalkStatus {
	attrs := []string{}
	switch node.Type {
	case Text:
		if r.Flags&Smartypants != 0 {
			var tmp bytes.Buffer
			escapeHTML(&tmp, node.Literal)
			r.sr.Process(w, tmp.Bytes())
		} else {
			if node.Parent.Type == Link {
				escLink(w, node.Literal)
			} else {
				escapeHTML(w, node.Literal)
			}
		}
	case Softbreak:
		r.cr(w)
		// TODO: make it configurable via out(renderer.softbreak)
	case Hardbreak:
		if r.Flags&UseXHTML == 0 {
			r.out(w, brTag)
		} else {
			r.out(w, brXHTMLTag)
		}
		r.cr(w)
	case Emph:
		if entering {
			r.out(w, emTag)
		} else {
			r.out(w, emCloseTag)
		}
	case Strong:
		if entering {
			r.out(w, strongTag)
		} else {
			r.out(w, strongCloseTag)
		}
	case Del:
		if entering {
			r.out(w, delTag)
		} else {
			r.out(w, delCloseTag)
		}
	case HTMLSpan:
		if r.Flags&SkipHTML != 0 {
			break
		}
		r.out(w, node.Literal)
	case Link:
		// mark it but don't link it if it is not a safe link: no smartypants
		dest := node.LinkData.Destination
		if needSkipLink(r.Flags, dest) {
			if entering {
				r.out(w, ttTag)
			} else {
				r.out(w, ttCloseTag)
			}
		} else {
			if entering {
				dest = r.addAbsPrefix(dest)
				var hrefBuf bytes.Buffer
				hrefBuf.WriteString("href=\"")
				escLink(&hrefBuf, dest)
				hrefBuf.WriteByte('"')
				attrs = append(attrs, hrefBuf.String())
				if node.NoteID != 0 {
					r.out(w, footnoteRef(r.FootnoteAnchorPrefix, node))
					break
				}
				attrs = appendLinkAttrs(attrs, r.Flags, dest)
				if len(node.LinkData.Title) > 0 {
					var titleBuff bytes.Buffer
					titleBuff.WriteString("title=\"")
					escapeHTML(&titleBuff, node.LinkData.Title)
					titleBuff.WriteByte('"')
					attrs = append(attrs, titleBuff.String())
				}
				r.tag(w, aTag, attrs)
			} else {
				if node.NoteID != 0 {
					break
				}
				r.out(w, aCloseTag)
			}
		}
	case Image:
		if r.Flags&SkipImages != 0 {
			return SkipChildren
		}
		if entering {
			dest := node.LinkData.Destination
			dest = r.addAbsPrefix(dest)
			if r.disableTags == 0 {
				//if options.safe && potentiallyUnsafe(dest) {
				//out(w, `<img src="" alt="`)
				//} else {
				r.out(w, []byte(`<img src="`))
				escLink(w, dest)
				r.out(w, []byte(`" alt="`))
				//}
			}
			r.disableTags++
		} else {
			r.disableTags--
			if r.disableTags == 0 {
				if node.LinkData.Title != nil {
					r.out(w, []byte(`" title="`))
					escapeHTML(w, node.LinkData.Title)
				}
				r.out(w, []byte(`" />`))
			}
		}
	case Code:
		r.out(w, codeTag)
		escapeHTML(w, node.Literal)
		r.out(w, codeCloseTag)
	case Document:
		break
	case Paragraph:
		if skipParagraphTags(node) {
			break
		}
		if entering {
			// TODO: untangle this clusterfuck about when the newlines need
			// to be added and when not.
			if node.Prev != nil {
				switch node.Prev.Type {
				case HTMLBlock, List, Paragraph, Heading, CodeBlock, BlockQuote, HorizontalRule:
					r.cr(w)
				}
			}
			if node.Parent.Type == BlockQuote && node.Prev == nil {
				r.cr(w)
			}
			r.out(w, pTag)
		} else {
			r.out(w, pCloseTag)
			if !(node.Parent.Type == Item && node.Next == nil) {
				r.cr(w)
			}
		}
	case BlockQuote:
		if entering {
			r.cr(w)
			r.out(w, blockquoteTag)
		} else {
			r.out(w, blockquoteCloseTag)
			r.cr(w)
		}
	case HTMLBlock:
		if r.Flags&SkipHTML != 0 {
			break
		}
		r.cr(w)
		r.out(w, node.Literal)
		r.cr(w)
	case Heading:
		headingLevel := r.HTMLRendererParameters.HeadingLevelOffset + node.Level
		openTag, closeTag := headingTagsFromLevel(headingLevel)
		if entering {
			if node.IsTitleblock {
				attrs = append(attrs, `class="title"`)
			}
			if node.HeadingID != "" {
				id := r.ensureUniqueHeadingID(node.HeadingID)
				if r.HeadingIDPrefix != "" {
					id = r.HeadingIDPrefix + id
				}
				if r.HeadingIDSuffix != "" {
					id = id + r.HeadingIDSuffix
				}
				attrs = append(attrs, fmt.Sprintf(`id="%s"`, id))
			}
			r.cr(w)
			r.tag(w, openTag, attrs)
		} else {
			r.out(w, closeTag)
			if !(node.Parent.Type == Item && node.Next == nil) {
				r.cr(w)
			}
		}
	case HorizontalRule:
		r.cr(w)
		r.outHRTag(w)
		r.cr(w)
	case List:
		openTag := ulTag
		closeTag := ulCloseTag
		if node.ListFlags&ListTypeOrdered != 0 {
			openTag = olTag
			closeTag = olCloseTag
		}
		if node.ListFlags&ListTypeDefinition != 0 {
			openTag = dlTag
			closeTag = dlCloseTag
		}
		if entering {
			if node.IsFootnotesList {
				r.out(w, footnotesDivBytes)
				r.outHRTag(w)
				r.cr(w)
			}
			r.cr(w)
			if node.Parent.Type == Item && node.Parent.Parent.Tight {
				r.cr(w)
			}
			r.tag(w, openTag[:len(openTag)-1], attrs)
			r.cr(w)
		} else {
			r.out(w, closeTag)
			//cr(w)
			//if node.parent.Type != Item {
			//	cr(w)
			//}
			if node.Parent.Type == Item && node.Next != nil {
				r.cr(w)
			}
			if node.Parent.Type == Document || node.Parent.Type == BlockQuote {
				r.cr(w)
			}
			if node.IsFootnotesList {
				r.out(w, footnotesCloseDivBytes)
			}
		}
	case Item:
		openTag := liTag
		closeTag := liCloseTag
		if node.ListFlags&ListTypeDefinition != 0 {
			openTag = ddTag
			closeTag = ddCloseTag
		}
		if node.ListFlags&ListTypeTerm != 0 {
			openTag = dtTag
			closeTag = dtCloseTag
		}
		if entering {
			if itemOpenCR(node) {
				r.cr(w)
			}
			if node.ListData.RefLink != nil {
				slug := slugify(node.ListData.RefLink)
				r.out(w, footnoteItem(r.FootnoteAnchorPrefix, slug))
				break
			}
			r.out(w, openTag)
		} else {
			if node.ListData.RefLink != nil {
				slug := slugify(node.ListData.RefLink)
				if r.Flags&FootnoteReturnLinks != 0 {
					r.out(w, footnoteReturnLink(r.FootnoteAnchorPrefix, r.FootnoteReturnLinkContents, slug))
				}
			}
			r.out(w, closeTag)
			r.cr(w)
		}
	case CodeBlock:
		attrs = appendLanguageAttr(attrs, node.Info)
		r.cr(w)
		r.out(w, preTag)
		r.tag(w, codeTag[:len(codeTag)-1], attrs)
		escapeHTML(w, node.Literal)
		r.out(w, codeCloseTag)
		r.out(w, preCloseTag)
		if node.Parent.Type != Item {
			r.cr(w)
		}
	case Table:
		if entering {
			r.cr(w)
			r.out(w, tableTag)
		} else {
			r.out(w, tableCloseTag)
			r.cr(w)
		}
	case TableCell:
		openTag := tdTag
		closeTag := tdCloseTag
		if node.IsHeader {
			openTag = thTag
			closeTag = thCloseTag
		}
		if entering {
			align := cellAlignment(node.Align)
			if align != "" {
				attrs = append(attrs, fmt.Sprintf(`align="%s"`, align))
			}
			if node.Prev == nil {
				r.cr(w)
			}
			r.tag(w, openTag, attrs)
		} else {
			r.out(w, closeTag)
			r.cr(w)
		}
	case TableHead:
		if entering {
			r.cr(w)
			r.out(w, theadTag)
		} else {
			r.out(w, theadCloseTag)
			r.cr(w)
		}
	case TableBody:
		if entering {
			r.cr(w)
			r.out(w, tbodyTag)
			// XXX: this is to adhere to a rather silly test. Should fix test.
			if node.FirstChild == nil {
				r.cr(w)
			}
		} else {
			r.out(w, tbodyCloseTag)
			r.cr(w)
		}
	case TableRow:
		if entering {
			r.cr(w)
			r.out(w, trTag)
		} else {
			r.out(w, trCloseTag)
			r.cr(w)
		}
	default:
		panic("Unknown node type " + node.Type.String())
	}
	return GoToNext
}

// RenderHeader writes HTML document preamble and TOC if requested.
func (r *HTMLRenderer) RenderHeader(w io.Writer, ast *Node) {
	r.writeDocumentHeader(w)
	if r.Flags&TOC != 0 {
		r.writeTOC(w, ast)
	}
}

// RenderFooter writes HTML document footer.
func (r *HTMLRenderer) RenderFooter(w io.Writer, ast *Node) {
	if r.Flags&CompletePage == 0 {
		return
	}
	io.WriteString(w, "\n</body>\n</html>\n")
}

func (r *HTMLRenderer) writeDocumentHeader(w io.Writer) {
	if r.Flags&CompletePage == 0 {
		return
	}
	ending := ""
	if r.Flags&UseXHTML != 0 {
		io.WriteString(w, "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" ")
		io.WriteString(w, "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n")
		io.WriteString(w, "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n")
		ending = " /"
	} else {
		io.WriteString(w, "<!DOCTYPE html>\n")
		io.WriteString(w, "<html>\n")
	}
	io.WriteString(w, "<head>\n")
	io.WriteString(w, "  <title>")
	if r.Flags&Smartypants != 0 {
		r.sr.Process(w, []byte(r.Title))
	} else {
		escapeHTML(w, []byte(r.Title))
	}
	io.WriteString(w, "</title>\n")
	io.WriteString(w, "  <meta name=\"GENERATOR\" content=\"Blackfriday Markdown Processor v")
	io.WriteString(w, Version)
	io.WriteString(w, "\"")
	io.WriteString(w, ending)
	io.WriteString(w, ">\n")
	io.WriteString(w, "  <meta charset=\"utf-8\"")
	io.WriteString(w, ending)
	io.WriteString(w, ">\n")
	if r.CSS != "" {
		io.WriteString(w, "  <link rel=\"stylesheet\" type=\"text/css\" href=\"")
		escapeHTML(w, []byte(r.CSS))
		io.WriteString(w, "\"")
		io.WriteString(w, ending)
		io.WriteString(w, ">\n")
	}
	if r.Icon != "" {
		io.WriteString(w, "  <link rel=\"icon\" type=\"image/x-icon\" href=\"")
		escapeHTML(w, []byte(r.Icon))
		io.WriteString(w, "\"")
		io.WriteString(w, ending)
		io.WriteString(w, ">\n")
	}
	io.WriteString(w, "</head>\n")
	io.WriteString(w, "<body>\n\n")
}

func (r *HTMLRenderer) writeTOC(w io.Writer, ast *Node) {
	buf := bytes.Buffer{}

	inHeading := false
	tocLevel := 0
	headingCount := 0

	ast.Walk(func(node *Node, entering bool) WalkStatus {
		if node.Type == Heading && !node.HeadingData.IsTitleblock {
			inHeading = entering
			if entering {
				node.HeadingID = fmt.Sprintf("toc_%d", headingCount)
				if node.Level == tocLevel {
					buf.WriteString("</li>\n\n<li>")
				} else if node.Level < tocLevel {
					for node.Level < tocLevel {
						tocLevel--
						buf.WriteString("</li>\n</ul>")
					}
					buf.WriteString("</li>\n\n<li>")
				} else {
					for node.Level > tocLevel {
						tocLevel++
						buf.WriteString("\n<ul>\n<li>")
					}
				}

				fmt.Fprintf(&buf, `<a href="#toc_%d">`, headingCount)
				headingCount++
			} else {
				buf.WriteString("</a>")
			}
			return GoToNext
		}

		if inHeading {
			return r.RenderNode(&buf, node, entering)
		}

		return GoToNext
	})

	for ; tocLevel > 0; tocLevel-- {
		buf.WriteString("</li>\n</ul>")
	}

	if buf.Len() > 0 {
		io.WriteString(w, "<nav>\n")
		w.Write(buf.Bytes())
		io.WriteString(w, "\n\n</nav>\n")
	}
	r.lastOutputLen = buf.Len()
}
