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
	"regexp"
	"strconv"
	"strings"
)

// Html renderer configuration options.
const (
	HTML_SKIP_HTML                 = 1 << iota // skip preformatted HTML blocks
	HTML_SKIP_STYLE                            // skip embedded <style> elements
	HTML_SKIP_IMAGES                           // skip embedded images
	HTML_SKIP_LINKS                            // skip all links
	HTML_SAFELINK                              // only link to trusted protocols
	HTML_NOFOLLOW_LINKS                        // only link with rel="nofollow"
	HTML_NOREFERRER_LINKS                      // only link with rel="noreferrer"
	HTML_HREF_TARGET_BLANK                     // add a blank target
	HTML_TOC                                   // generate a table of contents
	HTML_OMIT_CONTENTS                         // skip the main contents (for a standalone table of contents)
	HTML_COMPLETE_PAGE                         // generate a complete HTML page
	HTML_USE_XHTML                             // generate XHTML output instead of HTML
	HTML_USE_SMARTYPANTS                       // enable smart punctuation substitutions
	HTML_SMARTYPANTS_FRACTIONS                 // enable smart fractions (with HTML_USE_SMARTYPANTS)
	HTML_SMARTYPANTS_DASHES                    // enable smart dashes (with HTML_USE_SMARTYPANTS)
	HTML_SMARTYPANTS_LATEX_DASHES              // enable LaTeX-style dashes (with HTML_USE_SMARTYPANTS and HTML_SMARTYPANTS_DASHES)
	HTML_SMARTYPANTS_ANGLED_QUOTES             // enable angled double quotes (with HTML_USE_SMARTYPANTS) for double quotes rendering
	HTML_SMARTYPANTS_QUOTES_NBSP               // enable "French guillemets" (with HTML_USE_SMARTYPANTS)
	HTML_FOOTNOTE_RETURN_LINKS                 // generate a link at the end of a footnote to return to the source
)

var (
	alignments = []string{
		"left",
		"right",
		"center",
	}

	// TODO: improve this regexp to catch all possible entities:
	htmlEntity = regexp.MustCompile(`&[a-z]{2,5};`)
)

type HtmlRendererParameters struct {
	// Prepend this text to each relative URL.
	AbsolutePrefix string
	// Add this text to each footnote anchor, to ensure uniqueness.
	FootnoteAnchorPrefix string
	// Show this text inside the <a> tag for a footnote return link, if the
	// HTML_FOOTNOTE_RETURN_LINKS flag is enabled. If blank, the string
	// <sup>[return]</sup> is used.
	FootnoteReturnLinkContents string
	// If set, add this text to the front of each Header ID, to ensure
	// uniqueness.
	HeaderIDPrefix string
	// If set, add this text to the back of each Header ID, to ensure uniqueness.
	HeaderIDSuffix string
}

// Html is a type that implements the Renderer interface for HTML output.
//
// Do not create this directly, instead use the HtmlRenderer function.
type Html struct {
	flags    int    // HTML_* options
	closeTag string // how to end singleton tags: either " />" or ">"
	title    string // document title
	css      string // optional css file url (used with HTML_COMPLETE_PAGE)

	parameters HtmlRendererParameters

	// table of contents data
	tocMarker    int
	headerCount  int
	currentLevel int
	toc          *bytes.Buffer

	// Track header IDs to prevent ID collision in a single generation.
	headerIDs map[string]int

	smartypants *smartypantsRenderer
}

const (
	xhtmlClose = " />"
	htmlClose  = ">"
)

// HtmlRenderer creates and configures an Html object, which
// satisfies the Renderer interface.
//
// flags is a set of HTML_* options ORed together.
// title is the title of the document, and css is a URL for the document's
// stylesheet.
// title and css are only used when HTML_COMPLETE_PAGE is selected.
func HtmlRenderer(flags int, title string, css string) Renderer {
	return HtmlRendererWithParameters(flags, title, css, HtmlRendererParameters{})
}

func HtmlRendererWithParameters(flags int, title string,
	css string, renderParameters HtmlRendererParameters) Renderer {
	// configure the rendering engine
	closeTag := htmlClose
	if flags&HTML_USE_XHTML != 0 {
		closeTag = xhtmlClose
	}

	if renderParameters.FootnoteReturnLinkContents == "" {
		renderParameters.FootnoteReturnLinkContents = `<sup>[return]</sup>`
	}

	return &Html{
		flags:      flags,
		closeTag:   closeTag,
		title:      title,
		css:        css,
		parameters: renderParameters,

		headerCount:  0,
		currentLevel: 0,
		toc:          new(bytes.Buffer),

		headerIDs: make(map[string]int),

		smartypants: smartypants(flags),
	}
}

// Using if statements is a bit faster than a switch statement. As the compiler
// improves, this should be unnecessary this is only worthwhile because
// attrEscape is the single largest CPU user in normal use.
// Also tried using map, but that gave a ~3x slowdown.
func escapeSingleChar(char byte) (string, bool) {
	if char == '"' {
		return "&quot;", true
	}
	if char == '&' {
		return "&amp;", true
	}
	if char == '<' {
		return "&lt;", true
	}
	if char == '>' {
		return "&gt;", true
	}
	return "", false
}

func attrEscape(out *bytes.Buffer, src []byte) {
	org := 0
	for i, ch := range src {
		if entity, ok := escapeSingleChar(ch); ok {
			if i > org {
				// copy all the normal characters since the last escape
				out.Write(src[org:i])
			}
			org = i + 1
			out.WriteString(entity)
		}
	}
	if org < len(src) {
		out.Write(src[org:])
	}
}

func entityEscapeWithSkip(out *bytes.Buffer, src []byte, skipRanges [][]int) {
	end := 0
	for _, rang := range skipRanges {
		attrEscape(out, src[end:rang[0]])
		out.Write(src[rang[0]:rang[1]])
		end = rang[1]
	}
	attrEscape(out, src[end:])
}

func (options *Html) GetFlags() int {
	return options.flags
}

func (options *Html) TitleBlock(out *bytes.Buffer, text []byte) {
	text = bytes.TrimPrefix(text, []byte("% "))
	text = bytes.Replace(text, []byte("\n% "), []byte("\n"), -1)
	out.WriteString("<h1 class=\"title\">")
	out.Write(text)
	out.WriteString("\n</h1>")
}

func (options *Html) Header(out *bytes.Buffer, text func() bool, level int, id string) {
	marker := out.Len()
	doubleSpace(out)

	if id == "" && options.flags&HTML_TOC != 0 {
		id = fmt.Sprintf("toc_%d", options.headerCount)
	}

	if id != "" {
		id = options.ensureUniqueHeaderID(id)

		if options.parameters.HeaderIDPrefix != "" {
			id = options.parameters.HeaderIDPrefix + id
		}

		if options.parameters.HeaderIDSuffix != "" {
			id = id + options.parameters.HeaderIDSuffix
		}

		out.WriteString(fmt.Sprintf("<h%d id=\"%s\">", level, id))
	} else {
		out.WriteString(fmt.Sprintf("<h%d>", level))
	}

	tocMarker := out.Len()
	if !text() {
		out.Truncate(marker)
		return
	}

	// are we building a table of contents?
	if options.flags&HTML_TOC != 0 {
		options.TocHeaderWithAnchor(out.Bytes()[tocMarker:], level, id)
	}

	out.WriteString(fmt.Sprintf("</h%d>\n", level))
}

func (options *Html) BlockHtml(out *bytes.Buffer, text []byte) {
	if options.flags&HTML_SKIP_HTML != 0 {
		return
	}

	doubleSpace(out)
	out.Write(text)
	out.WriteByte('\n')
}

func (options *Html) HRule(out *bytes.Buffer) {
	doubleSpace(out)
	out.WriteString("<hr")
	out.WriteString(options.closeTag)
	out.WriteByte('\n')
}

func (options *Html) BlockCode(out *bytes.Buffer, text []byte, info string) {
	doubleSpace(out)

	endOfLang := strings.IndexAny(info, "\t ")
	if endOfLang < 0 {
		endOfLang = len(info)
	}
	lang := info[:endOfLang]
	if len(lang) == 0 || lang == "." {
		out.WriteString("<pre><code>")
	} else {
		out.WriteString("<pre><code class=\"language-")
		attrEscape(out, []byte(lang))
		out.WriteString("\">")
	}
	attrEscape(out, text)
	out.WriteString("</code></pre>\n")
}

func (options *Html) BlockQuote(out *bytes.Buffer, text []byte) {
	doubleSpace(out)
	out.WriteString("<blockquote>\n")
	out.Write(text)
	out.WriteString("</blockquote>\n")
}

func (options *Html) Table(out *bytes.Buffer, header []byte, body []byte, columnData []int) {
	doubleSpace(out)
	out.WriteString("<table>\n<thead>\n")
	out.Write(header)
	out.WriteString("</thead>\n\n<tbody>\n")
	out.Write(body)
	out.WriteString("</tbody>\n</table>\n")
}

func (options *Html) TableRow(out *bytes.Buffer, text []byte) {
	doubleSpace(out)
	out.WriteString("<tr>\n")
	out.Write(text)
	out.WriteString("\n</tr>\n")
}

func (options *Html) TableHeaderCell(out *bytes.Buffer, text []byte, align int) {
	doubleSpace(out)
	switch align {
	case TABLE_ALIGNMENT_LEFT:
		out.WriteString("<th align=\"left\">")
	case TABLE_ALIGNMENT_RIGHT:
		out.WriteString("<th align=\"right\">")
	case TABLE_ALIGNMENT_CENTER:
		out.WriteString("<th align=\"center\">")
	default:
		out.WriteString("<th>")
	}

	out.Write(text)
	out.WriteString("</th>")
}

func (options *Html) TableCell(out *bytes.Buffer, text []byte, align int) {
	doubleSpace(out)
	switch align {
	case TABLE_ALIGNMENT_LEFT:
		out.WriteString("<td align=\"left\">")
	case TABLE_ALIGNMENT_RIGHT:
		out.WriteString("<td align=\"right\">")
	case TABLE_ALIGNMENT_CENTER:
		out.WriteString("<td align=\"center\">")
	default:
		out.WriteString("<td>")
	}

	out.Write(text)
	out.WriteString("</td>")
}

func (options *Html) Footnotes(out *bytes.Buffer, text func() bool) {
	out.WriteString("<div class=\"footnotes\">\n")
	options.HRule(out)
	options.List(out, text, LIST_TYPE_ORDERED)
	out.WriteString("</div>\n")
}

func (options *Html) FootnoteItem(out *bytes.Buffer, name, text []byte, flags int) {
	if flags&LIST_ITEM_CONTAINS_BLOCK != 0 || flags&LIST_ITEM_BEGINNING_OF_LIST != 0 {
		doubleSpace(out)
	}
	slug := slugify(name)
	out.WriteString(`<li id="`)
	out.WriteString(`fn:`)
	out.WriteString(options.parameters.FootnoteAnchorPrefix)
	out.Write(slug)
	out.WriteString(`">`)
	out.Write(text)
	if options.flags&HTML_FOOTNOTE_RETURN_LINKS != 0 {
		out.WriteString(` <a class="footnote-return" href="#`)
		out.WriteString(`fnref:`)
		out.WriteString(options.parameters.FootnoteAnchorPrefix)
		out.Write(slug)
		out.WriteString(`">`)
		out.WriteString(options.parameters.FootnoteReturnLinkContents)
		out.WriteString(`</a>`)
	}
	out.WriteString("</li>\n")
}

func (options *Html) List(out *bytes.Buffer, text func() bool, flags int) {
	marker := out.Len()
	doubleSpace(out)

	if flags&LIST_TYPE_DEFINITION != 0 {
		out.WriteString("<dl>")
	} else if flags&LIST_TYPE_ORDERED != 0 {
		out.WriteString("<ol>")
	} else {
		out.WriteString("<ul>")
	}
	if !text() {
		out.Truncate(marker)
		return
	}
	if flags&LIST_TYPE_DEFINITION != 0 {
		out.WriteString("</dl>\n")
	} else if flags&LIST_TYPE_ORDERED != 0 {
		out.WriteString("</ol>\n")
	} else {
		out.WriteString("</ul>\n")
	}
}

func (options *Html) ListItem(out *bytes.Buffer, text []byte, flags int) {
	if (flags&LIST_ITEM_CONTAINS_BLOCK != 0 && flags&LIST_TYPE_DEFINITION == 0) ||
		flags&LIST_ITEM_BEGINNING_OF_LIST != 0 {
		doubleSpace(out)
	}
	if flags&LIST_TYPE_TERM != 0 {
		out.WriteString("<dt>")
	} else if flags&LIST_TYPE_DEFINITION != 0 {
		out.WriteString("<dd>")
	} else {
		out.WriteString("<li>")
	}
	out.Write(text)
	if flags&LIST_TYPE_TERM != 0 {
		out.WriteString("</dt>\n")
	} else if flags&LIST_TYPE_DEFINITION != 0 {
		out.WriteString("</dd>\n")
	} else {
		out.WriteString("</li>\n")
	}
}

func (options *Html) Paragraph(out *bytes.Buffer, text func() bool) {
	marker := out.Len()
	doubleSpace(out)

	out.WriteString("<p>")
	if !text() {
		out.Truncate(marker)
		return
	}
	out.WriteString("</p>\n")
}

func (options *Html) AutoLink(out *bytes.Buffer, link []byte, kind int) {
	skipRanges := htmlEntity.FindAllIndex(link, -1)
	if options.flags&HTML_SAFELINK != 0 && !isSafeLink(link) && kind != LINK_TYPE_EMAIL {
		// mark it but don't link it if it is not a safe link: no smartypants
		out.WriteString("<tt>")
		entityEscapeWithSkip(out, link, skipRanges)
		out.WriteString("</tt>")
		return
	}

	out.WriteString("<a href=\"")
	if kind == LINK_TYPE_EMAIL {
		out.WriteString("mailto:")
	} else {
		options.maybeWriteAbsolutePrefix(out, link)
	}

	entityEscapeWithSkip(out, link, skipRanges)

	var relAttrs []string
	if options.flags&HTML_NOFOLLOW_LINKS != 0 && !isRelativeLink(link) {
		relAttrs = append(relAttrs, "nofollow")
	}
	if options.flags&HTML_NOREFERRER_LINKS != 0 && !isRelativeLink(link) {
		relAttrs = append(relAttrs, "noreferrer")
	}
	if len(relAttrs) > 0 {
		out.WriteString(fmt.Sprintf("\" rel=\"%s", strings.Join(relAttrs, " ")))
	}

	// blank target only add to external link
	if options.flags&HTML_HREF_TARGET_BLANK != 0 && !isRelativeLink(link) {
		out.WriteString("\" target=\"_blank")
	}

	out.WriteString("\">")

	// Pretty print: if we get an email address as
	// an actual URI, e.g. `mailto:foo@bar.com`, we don't
	// want to print the `mailto:` prefix
	switch {
	case bytes.HasPrefix(link, []byte("mailto://")):
		attrEscape(out, link[len("mailto://"):])
	case bytes.HasPrefix(link, []byte("mailto:")):
		attrEscape(out, link[len("mailto:"):])
	default:
		entityEscapeWithSkip(out, link, skipRanges)
	}

	out.WriteString("</a>")
}

func (options *Html) CodeSpan(out *bytes.Buffer, text []byte) {
	out.WriteString("<code>")
	attrEscape(out, text)
	out.WriteString("</code>")
}

func (options *Html) DoubleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("<strong>")
	out.Write(text)
	out.WriteString("</strong>")
}

func (options *Html) Emphasis(out *bytes.Buffer, text []byte) {
	if len(text) == 0 {
		return
	}
	out.WriteString("<em>")
	out.Write(text)
	out.WriteString("</em>")
}

func (options *Html) maybeWriteAbsolutePrefix(out *bytes.Buffer, link []byte) {
	if options.parameters.AbsolutePrefix != "" && isRelativeLink(link) && link[0] != '.' {
		out.WriteString(options.parameters.AbsolutePrefix)
		if link[0] != '/' {
			out.WriteByte('/')
		}
	}
}

func (options *Html) Image(out *bytes.Buffer, link []byte, title []byte, alt []byte) {
	if options.flags&HTML_SKIP_IMAGES != 0 {
		return
	}

	out.WriteString("<img src=\"")
	options.maybeWriteAbsolutePrefix(out, link)
	attrEscape(out, link)
	out.WriteString("\" alt=\"")
	if len(alt) > 0 {
		attrEscape(out, alt)
	}
	if len(title) > 0 {
		out.WriteString("\" title=\"")
		attrEscape(out, title)
	}

	out.WriteByte('"')
	out.WriteString(options.closeTag)
}

func (options *Html) LineBreak(out *bytes.Buffer) {
	out.WriteString("<br")
	out.WriteString(options.closeTag)
	out.WriteByte('\n')
}

func (options *Html) Link(out *bytes.Buffer, link []byte, title []byte, content []byte) {
	if options.flags&HTML_SKIP_LINKS != 0 {
		// write the link text out but don't link it, just mark it with typewriter font
		out.WriteString("<tt>")
		attrEscape(out, content)
		out.WriteString("</tt>")
		return
	}

	if options.flags&HTML_SAFELINK != 0 && !isSafeLink(link) {
		// write the link text out but don't link it, just mark it with typewriter font
		out.WriteString("<tt>")
		attrEscape(out, content)
		out.WriteString("</tt>")
		return
	}

	out.WriteString("<a href=\"")
	options.maybeWriteAbsolutePrefix(out, link)
	attrEscape(out, link)
	if len(title) > 0 {
		out.WriteString("\" title=\"")
		attrEscape(out, title)
	}
	var relAttrs []string
	if options.flags&HTML_NOFOLLOW_LINKS != 0 && !isRelativeLink(link) {
		relAttrs = append(relAttrs, "nofollow")
	}
	if options.flags&HTML_NOREFERRER_LINKS != 0 && !isRelativeLink(link) {
		relAttrs = append(relAttrs, "noreferrer")
	}
	if len(relAttrs) > 0 {
		out.WriteString(fmt.Sprintf("\" rel=\"%s", strings.Join(relAttrs, " ")))
	}

	// blank target only add to external link
	if options.flags&HTML_HREF_TARGET_BLANK != 0 && !isRelativeLink(link) {
		out.WriteString("\" target=\"_blank")
	}

	out.WriteString("\">")
	out.Write(content)
	out.WriteString("</a>")
	return
}

func (options *Html) RawHtmlTag(out *bytes.Buffer, text []byte) {
	if options.flags&HTML_SKIP_HTML != 0 {
		return
	}
	if options.flags&HTML_SKIP_STYLE != 0 && isHtmlTag(text, "style") {
		return
	}
	if options.flags&HTML_SKIP_LINKS != 0 && isHtmlTag(text, "a") {
		return
	}
	if options.flags&HTML_SKIP_IMAGES != 0 && isHtmlTag(text, "img") {
		return
	}
	out.Write(text)
}

func (options *Html) TripleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("<strong><em>")
	out.Write(text)
	out.WriteString("</em></strong>")
}

func (options *Html) StrikeThrough(out *bytes.Buffer, text []byte) {
	out.WriteString("<del>")
	out.Write(text)
	out.WriteString("</del>")
}

func (options *Html) FootnoteRef(out *bytes.Buffer, ref []byte, id int) {
	slug := slugify(ref)
	out.WriteString(`<sup class="footnote-ref" id="`)
	out.WriteString(`fnref:`)
	out.WriteString(options.parameters.FootnoteAnchorPrefix)
	out.Write(slug)
	out.WriteString(`"><a href="#`)
	out.WriteString(`fn:`)
	out.WriteString(options.parameters.FootnoteAnchorPrefix)
	out.Write(slug)
	out.WriteString(`">`)
	out.WriteString(strconv.Itoa(id))
	out.WriteString(`</a></sup>`)
}

func (options *Html) Entity(out *bytes.Buffer, entity []byte) {
	out.Write(entity)
}

func (options *Html) NormalText(out *bytes.Buffer, text []byte) {
	if options.flags&HTML_USE_SMARTYPANTS != 0 {
		options.Smartypants(out, text)
	} else {
		attrEscape(out, text)
	}
}

func (options *Html) Smartypants(out *bytes.Buffer, text []byte) {
	smrt := smartypantsData{false, false}

	// first do normal entity escaping
	var escaped bytes.Buffer
	attrEscape(&escaped, text)
	text = escaped.Bytes()

	mark := 0
	for i := 0; i < len(text); i++ {
		if action := options.smartypants[text[i]]; action != nil {
			if i > mark {
				out.Write(text[mark:i])
			}

			previousChar := byte(0)
			if i > 0 {
				previousChar = text[i-1]
			}
			i += action(out, &smrt, previousChar, text[i:])
			mark = i + 1
		}
	}

	if mark < len(text) {
		out.Write(text[mark:])
	}
}

func (options *Html) DocumentHeader(out *bytes.Buffer) {
	if options.flags&HTML_COMPLETE_PAGE == 0 {
		return
	}

	ending := ""
	if options.flags&HTML_USE_XHTML != 0 {
		out.WriteString("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" ")
		out.WriteString("\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n")
		out.WriteString("<html xmlns=\"http://www.w3.org/1999/xhtml\">\n")
		ending = " /"
	} else {
		out.WriteString("<!DOCTYPE html>\n")
		out.WriteString("<html>\n")
	}
	out.WriteString("<head>\n")
	out.WriteString("  <title>")
	options.NormalText(out, []byte(options.title))
	out.WriteString("</title>\n")
	out.WriteString("  <meta name=\"GENERATOR\" content=\"Blackfriday Markdown Processor v")
	out.WriteString(VERSION)
	out.WriteString("\"")
	out.WriteString(ending)
	out.WriteString(">\n")
	out.WriteString("  <meta charset=\"utf-8\"")
	out.WriteString(ending)
	out.WriteString(">\n")
	if options.css != "" {
		out.WriteString("  <link rel=\"stylesheet\" type=\"text/css\" href=\"")
		attrEscape(out, []byte(options.css))
		out.WriteString("\"")
		out.WriteString(ending)
		out.WriteString(">\n")
	}
	out.WriteString("</head>\n")
	out.WriteString("<body>\n")

	options.tocMarker = out.Len()
}

func (options *Html) DocumentFooter(out *bytes.Buffer) {
	// finalize and insert the table of contents
	if options.flags&HTML_TOC != 0 {
		options.TocFinalize()

		// now we have to insert the table of contents into the document
		var temp bytes.Buffer

		// start by making a copy of everything after the document header
		temp.Write(out.Bytes()[options.tocMarker:])

		// now clear the copied material from the main output buffer
		out.Truncate(options.tocMarker)

		// corner case spacing issue
		if options.flags&HTML_COMPLETE_PAGE != 0 {
			out.WriteByte('\n')
		}

		// insert the table of contents
		out.WriteString("<nav>\n")
		out.Write(options.toc.Bytes())
		out.WriteString("</nav>\n")

		// corner case spacing issue
		if options.flags&HTML_COMPLETE_PAGE == 0 && options.flags&HTML_OMIT_CONTENTS == 0 {
			out.WriteByte('\n')
		}

		// write out everything that came after it
		if options.flags&HTML_OMIT_CONTENTS == 0 {
			out.Write(temp.Bytes())
		}
	}

	if options.flags&HTML_COMPLETE_PAGE != 0 {
		out.WriteString("\n</body>\n")
		out.WriteString("</html>\n")
	}

}

func (options *Html) TocHeaderWithAnchor(text []byte, level int, anchor string) {
	for level > options.currentLevel {
		switch {
		case bytes.HasSuffix(options.toc.Bytes(), []byte("</li>\n")):
			// this sublist can nest underneath a header
			size := options.toc.Len()
			options.toc.Truncate(size - len("</li>\n"))

		case options.currentLevel > 0:
			options.toc.WriteString("<li>")
		}
		if options.toc.Len() > 0 {
			options.toc.WriteByte('\n')
		}
		options.toc.WriteString("<ul>\n")
		options.currentLevel++
	}

	for level < options.currentLevel {
		options.toc.WriteString("</ul>")
		if options.currentLevel > 1 {
			options.toc.WriteString("</li>\n")
		}
		options.currentLevel--
	}

	options.toc.WriteString("<li><a href=\"#")
	if anchor != "" {
		options.toc.WriteString(anchor)
	} else {
		options.toc.WriteString("toc_")
		options.toc.WriteString(strconv.Itoa(options.headerCount))
	}
	options.toc.WriteString("\">")
	options.headerCount++

	options.toc.Write(text)

	options.toc.WriteString("</a></li>\n")
}

func (options *Html) TocHeader(text []byte, level int) {
	options.TocHeaderWithAnchor(text, level, "")
}

func (options *Html) TocFinalize() {
	for options.currentLevel > 1 {
		options.toc.WriteString("</ul></li>\n")
		options.currentLevel--
	}

	if options.currentLevel > 0 {
		options.toc.WriteString("</ul>\n")
	}
}

func isHtmlTag(tag []byte, tagname string) bool {
	found, _ := findHtmlTagPos(tag, tagname)
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

func findHtmlTagPos(tag []byte, tagname string) (bool, int) {
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
	if rightAngle > i {
		return true, rightAngle
	}

	return false, -1
}

func skipUntilChar(text []byte, start int, char byte) int {
	i := start
	for i < len(text) && text[i] != char {
		i++
	}
	return i
}

func skipSpace(tag []byte, i int) int {
	for i < len(tag) && isspace(tag[i]) {
		i++
	}
	return i
}

func skipChar(data []byte, start int, char byte) int {
	i := start
	for i < len(data) && data[i] == char {
		i++
	}
	return i
}

func doubleSpace(out *bytes.Buffer) {
	if out.Len() > 0 {
		out.WriteByte('\n')
	}
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

func (options *Html) ensureUniqueHeaderID(id string) string {
	for count, found := options.headerIDs[id]; found; count, found = options.headerIDs[id] {
		tmp := fmt.Sprintf("%s-%d", id, count+1)

		if _, tmpFound := options.headerIDs[tmp]; !tmpFound {
			options.headerIDs[id] = count + 1
			id = tmp
		} else {
			id = id + "-1"
		}
	}

	if _, found := options.headerIDs[id]; !found {
		options.headerIDs[id] = 0
	}

	return id
}
