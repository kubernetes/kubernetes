package md2man

import (
	"bytes"
	"fmt"
	"html"
	"strings"

	"github.com/russross/blackfriday"
)

type roffRenderer struct {
	ListCounters []int
}

// RoffRenderer creates a new blackfriday Renderer for generating roff documents
// from markdown
func RoffRenderer(flags int) blackfriday.Renderer {
	return &roffRenderer{}
}

func (r *roffRenderer) GetFlags() int {
	return 0
}

func (r *roffRenderer) TitleBlock(out *bytes.Buffer, text []byte) {
	out.WriteString(".TH ")

	splitText := bytes.Split(text, []byte("\n"))
	for i, line := range splitText {
		line = bytes.TrimPrefix(line, []byte("% "))
		if i == 0 {
			line = bytes.Replace(line, []byte("("), []byte("\" \""), 1)
			line = bytes.Replace(line, []byte(")"), []byte("\" \""), 1)
		}
		line = append([]byte("\""), line...)
		line = append(line, []byte("\" ")...)
		out.Write(line)
	}
	out.WriteString("\n")

	// disable hyphenation
	out.WriteString(".nh\n")
	// disable justification (adjust text to left margin only)
	out.WriteString(".ad l\n")
}

func (r *roffRenderer) BlockCode(out *bytes.Buffer, text []byte, lang string) {
	out.WriteString("\n.PP\n.RS\n\n.nf\n")
	escapeSpecialChars(out, text)
	out.WriteString("\n.fi\n.RE\n")
}

func (r *roffRenderer) BlockQuote(out *bytes.Buffer, text []byte) {
	out.WriteString("\n.PP\n.RS\n")
	out.Write(text)
	out.WriteString("\n.RE\n")
}

func (r *roffRenderer) BlockHtml(out *bytes.Buffer, text []byte) { // nolint: golint
	out.Write(text)
}

func (r *roffRenderer) Header(out *bytes.Buffer, text func() bool, level int, id string) {
	marker := out.Len()

	switch {
	case marker == 0:
		// This is the doc header
		out.WriteString(".TH ")
	case level == 1:
		out.WriteString("\n\n.SH ")
	case level == 2:
		out.WriteString("\n.SH ")
	default:
		out.WriteString("\n.SS ")
	}

	if !text() {
		out.Truncate(marker)
		return
	}
}

func (r *roffRenderer) HRule(out *bytes.Buffer) {
	out.WriteString("\n.ti 0\n\\l'\\n(.lu'\n")
}

func (r *roffRenderer) List(out *bytes.Buffer, text func() bool, flags int) {
	marker := out.Len()
	r.ListCounters = append(r.ListCounters, 1)
	out.WriteString("\n.RS\n")
	if !text() {
		out.Truncate(marker)
		return
	}
	r.ListCounters = r.ListCounters[:len(r.ListCounters)-1]
	out.WriteString("\n.RE\n")
}

func (r *roffRenderer) ListItem(out *bytes.Buffer, text []byte, flags int) {
	if flags&blackfriday.LIST_TYPE_ORDERED != 0 {
		out.WriteString(fmt.Sprintf(".IP \"%3d.\" 5\n", r.ListCounters[len(r.ListCounters)-1]))
		r.ListCounters[len(r.ListCounters)-1]++
	} else {
		out.WriteString(".IP \\(bu 2\n")
	}
	out.Write(text)
	out.WriteString("\n")
}

func (r *roffRenderer) Paragraph(out *bytes.Buffer, text func() bool) {
	marker := out.Len()
	out.WriteString("\n.PP\n")
	if !text() {
		out.Truncate(marker)
		return
	}
	if marker != 0 {
		out.WriteString("\n")
	}
}

func (r *roffRenderer) Table(out *bytes.Buffer, header []byte, body []byte, columnData []int) {
	out.WriteString("\n.TS\nallbox;\n")

	maxDelims := 0
	lines := strings.Split(strings.TrimRight(string(header), "\n")+"\n"+strings.TrimRight(string(body), "\n"), "\n")
	for _, w := range lines {
		curDelims := strings.Count(w, "\t")
		if curDelims > maxDelims {
			maxDelims = curDelims
		}
	}
	out.Write([]byte(strings.Repeat("l ", maxDelims+1) + "\n"))
	out.Write([]byte(strings.Repeat("l ", maxDelims+1) + ".\n"))
	out.Write(header)
	if len(header) > 0 {
		out.Write([]byte("\n"))
	}

	out.Write(body)
	out.WriteString("\n.TE\n")
}

func (r *roffRenderer) TableRow(out *bytes.Buffer, text []byte) {
	if out.Len() > 0 {
		out.WriteString("\n")
	}
	out.Write(text)
}

func (r *roffRenderer) TableHeaderCell(out *bytes.Buffer, text []byte, align int) {
	if out.Len() > 0 {
		out.WriteString("\t")
	}
	if len(text) == 0 {
		text = []byte{' '}
	}
	out.Write([]byte("\\fB\\fC" + string(text) + "\\fR"))
}

func (r *roffRenderer) TableCell(out *bytes.Buffer, text []byte, align int) {
	if out.Len() > 0 {
		out.WriteString("\t")
	}
	if len(text) > 30 {
		text = append([]byte("T{\n"), text...)
		text = append(text, []byte("\nT}")...)
	}
	if len(text) == 0 {
		text = []byte{' '}
	}
	out.Write(text)
}

func (r *roffRenderer) Footnotes(out *bytes.Buffer, text func() bool) {

}

func (r *roffRenderer) FootnoteItem(out *bytes.Buffer, name, text []byte, flags int) {

}

func (r *roffRenderer) AutoLink(out *bytes.Buffer, link []byte, kind int) {
	out.WriteString("\n\\[la]")
	out.Write(link)
	out.WriteString("\\[ra]")
}

func (r *roffRenderer) CodeSpan(out *bytes.Buffer, text []byte) {
	out.WriteString("\\fB\\fC")
	escapeSpecialChars(out, text)
	out.WriteString("\\fR")
}

func (r *roffRenderer) DoubleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\fB")
	out.Write(text)
	out.WriteString("\\fP")
}

func (r *roffRenderer) Emphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\fI")
	out.Write(text)
	out.WriteString("\\fP")
}

func (r *roffRenderer) Image(out *bytes.Buffer, link []byte, title []byte, alt []byte) {
}

func (r *roffRenderer) LineBreak(out *bytes.Buffer) {
	out.WriteString("\n.br\n")
}

func (r *roffRenderer) Link(out *bytes.Buffer, link []byte, title []byte, content []byte) {
	out.Write(content)
	r.AutoLink(out, link, 0)
}

func (r *roffRenderer) RawHtmlTag(out *bytes.Buffer, tag []byte) { // nolint: golint
	out.Write(tag)
}

func (r *roffRenderer) TripleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\s+2")
	out.Write(text)
	out.WriteString("\\s-2")
}

func (r *roffRenderer) StrikeThrough(out *bytes.Buffer, text []byte) {
}

func (r *roffRenderer) FootnoteRef(out *bytes.Buffer, ref []byte, id int) {

}

func (r *roffRenderer) Entity(out *bytes.Buffer, entity []byte) {
	out.WriteString(html.UnescapeString(string(entity)))
}

func (r *roffRenderer) NormalText(out *bytes.Buffer, text []byte) {
	escapeSpecialChars(out, text)
}

func (r *roffRenderer) DocumentHeader(out *bytes.Buffer) {
}

func (r *roffRenderer) DocumentFooter(out *bytes.Buffer) {
}

func needsBackslash(c byte) bool {
	for _, r := range []byte("-_&\\~") {
		if c == r {
			return true
		}
	}
	return false
}

func escapeSpecialChars(out *bytes.Buffer, text []byte) {
	for i := 0; i < len(text); i++ {
		// escape initial apostrophe or period
		if len(text) >= 1 && (text[0] == '\'' || text[0] == '.') {
			out.WriteString("\\&")
		}

		// directly copy normal characters
		org := i

		for i < len(text) && !needsBackslash(text[i]) {
			i++
		}
		if i > org {
			out.Write(text[org:i])
		}

		// escape a character
		if i >= len(text) {
			break
		}
		out.WriteByte('\\')
		out.WriteByte(text[i])
	}
}
