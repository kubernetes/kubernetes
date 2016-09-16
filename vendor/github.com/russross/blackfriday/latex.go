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
// LaTeX rendering backend
//
//

package blackfriday

import (
	"bytes"
)

// Latex is a type that implements the Renderer interface for LaTeX output.
//
// Do not create this directly, instead use the LatexRenderer function.
type Latex struct {
}

// LatexRenderer creates and configures a Latex object, which
// satisfies the Renderer interface.
//
// flags is a set of LATEX_* options ORed together (currently no such options
// are defined).
func LatexRenderer(flags int) Renderer {
	return &Latex{}
}

func (options *Latex) GetFlags() int {
	return 0
}

// render code chunks using verbatim, or listings if we have a language
func (options *Latex) BlockCode(out *bytes.Buffer, text []byte, lang string) {
	if lang == "" {
		out.WriteString("\n\\begin{verbatim}\n")
	} else {
		out.WriteString("\n\\begin{lstlisting}[language=")
		out.WriteString(lang)
		out.WriteString("]\n")
	}
	out.Write(text)
	if lang == "" {
		out.WriteString("\n\\end{verbatim}\n")
	} else {
		out.WriteString("\n\\end{lstlisting}\n")
	}
}

func (options *Latex) TitleBlock(out *bytes.Buffer, text []byte) {

}

func (options *Latex) BlockQuote(out *bytes.Buffer, text []byte) {
	out.WriteString("\n\\begin{quotation}\n")
	out.Write(text)
	out.WriteString("\n\\end{quotation}\n")
}

func (options *Latex) BlockHtml(out *bytes.Buffer, text []byte) {
	// a pretty lame thing to do...
	out.WriteString("\n\\begin{verbatim}\n")
	out.Write(text)
	out.WriteString("\n\\end{verbatim}\n")
}

func (options *Latex) Header(out *bytes.Buffer, text func() bool, level int, id string) {
	marker := out.Len()

	switch level {
	case 1:
		out.WriteString("\n\\section{")
	case 2:
		out.WriteString("\n\\subsection{")
	case 3:
		out.WriteString("\n\\subsubsection{")
	case 4:
		out.WriteString("\n\\paragraph{")
	case 5:
		out.WriteString("\n\\subparagraph{")
	case 6:
		out.WriteString("\n\\textbf{")
	}
	if !text() {
		out.Truncate(marker)
		return
	}
	out.WriteString("}\n")
}

func (options *Latex) HRule(out *bytes.Buffer) {
	out.WriteString("\n\\HRule\n")
}

func (options *Latex) List(out *bytes.Buffer, text func() bool, flags int) {
	marker := out.Len()
	if flags&LIST_TYPE_ORDERED != 0 {
		out.WriteString("\n\\begin{enumerate}\n")
	} else {
		out.WriteString("\n\\begin{itemize}\n")
	}
	if !text() {
		out.Truncate(marker)
		return
	}
	if flags&LIST_TYPE_ORDERED != 0 {
		out.WriteString("\n\\end{enumerate}\n")
	} else {
		out.WriteString("\n\\end{itemize}\n")
	}
}

func (options *Latex) ListItem(out *bytes.Buffer, text []byte, flags int) {
	out.WriteString("\n\\item ")
	out.Write(text)
}

func (options *Latex) Paragraph(out *bytes.Buffer, text func() bool) {
	marker := out.Len()
	out.WriteString("\n")
	if !text() {
		out.Truncate(marker)
		return
	}
	out.WriteString("\n")
}

func (options *Latex) Table(out *bytes.Buffer, header []byte, body []byte, columnData []int) {
	out.WriteString("\n\\begin{tabular}{")
	for _, elt := range columnData {
		switch elt {
		case TABLE_ALIGNMENT_LEFT:
			out.WriteByte('l')
		case TABLE_ALIGNMENT_RIGHT:
			out.WriteByte('r')
		default:
			out.WriteByte('c')
		}
	}
	out.WriteString("}\n")
	out.Write(header)
	out.WriteString(" \\\\\n\\hline\n")
	out.Write(body)
	out.WriteString("\n\\end{tabular}\n")
}

func (options *Latex) TableRow(out *bytes.Buffer, text []byte) {
	if out.Len() > 0 {
		out.WriteString(" \\\\\n")
	}
	out.Write(text)
}

func (options *Latex) TableHeaderCell(out *bytes.Buffer, text []byte, align int) {
	if out.Len() > 0 {
		out.WriteString(" & ")
	}
	out.Write(text)
}

func (options *Latex) TableCell(out *bytes.Buffer, text []byte, align int) {
	if out.Len() > 0 {
		out.WriteString(" & ")
	}
	out.Write(text)
}

// TODO: this
func (options *Latex) Footnotes(out *bytes.Buffer, text func() bool) {

}

func (options *Latex) FootnoteItem(out *bytes.Buffer, name, text []byte, flags int) {

}

func (options *Latex) AutoLink(out *bytes.Buffer, link []byte, kind int) {
	out.WriteString("\\href{")
	if kind == LINK_TYPE_EMAIL {
		out.WriteString("mailto:")
	}
	out.Write(link)
	out.WriteString("}{")
	out.Write(link)
	out.WriteString("}")
}

func (options *Latex) CodeSpan(out *bytes.Buffer, text []byte) {
	out.WriteString("\\texttt{")
	escapeSpecialChars(out, text)
	out.WriteString("}")
}

func (options *Latex) DoubleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\textbf{")
	out.Write(text)
	out.WriteString("}")
}

func (options *Latex) Emphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\textit{")
	out.Write(text)
	out.WriteString("}")
}

func (options *Latex) Image(out *bytes.Buffer, link []byte, title []byte, alt []byte) {
	if bytes.HasPrefix(link, []byte("http://")) || bytes.HasPrefix(link, []byte("https://")) {
		// treat it like a link
		out.WriteString("\\href{")
		out.Write(link)
		out.WriteString("}{")
		out.Write(alt)
		out.WriteString("}")
	} else {
		out.WriteString("\\includegraphics{")
		out.Write(link)
		out.WriteString("}")
	}
}

func (options *Latex) LineBreak(out *bytes.Buffer) {
	out.WriteString(" \\\\\n")
}

func (options *Latex) Link(out *bytes.Buffer, link []byte, title []byte, content []byte) {
	out.WriteString("\\href{")
	out.Write(link)
	out.WriteString("}{")
	out.Write(content)
	out.WriteString("}")
}

func (options *Latex) RawHtmlTag(out *bytes.Buffer, tag []byte) {
}

func (options *Latex) TripleEmphasis(out *bytes.Buffer, text []byte) {
	out.WriteString("\\textbf{\\textit{")
	out.Write(text)
	out.WriteString("}}")
}

func (options *Latex) StrikeThrough(out *bytes.Buffer, text []byte) {
	out.WriteString("\\sout{")
	out.Write(text)
	out.WriteString("}")
}

// TODO: this
func (options *Latex) FootnoteRef(out *bytes.Buffer, ref []byte, id int) {

}

func needsBackslash(c byte) bool {
	for _, r := range []byte("_{}%$&\\~#") {
		if c == r {
			return true
		}
	}
	return false
}

func escapeSpecialChars(out *bytes.Buffer, text []byte) {
	for i := 0; i < len(text); i++ {
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

func (options *Latex) Entity(out *bytes.Buffer, entity []byte) {
	// TODO: convert this into a unicode character or something
	out.Write(entity)
}

func (options *Latex) NormalText(out *bytes.Buffer, text []byte) {
	escapeSpecialChars(out, text)
}

// header and footer
func (options *Latex) DocumentHeader(out *bytes.Buffer) {
	out.WriteString("\\documentclass{article}\n")
	out.WriteString("\n")
	out.WriteString("\\usepackage{graphicx}\n")
	out.WriteString("\\usepackage{listings}\n")
	out.WriteString("\\usepackage[margin=1in]{geometry}\n")
	out.WriteString("\\usepackage[utf8]{inputenc}\n")
	out.WriteString("\\usepackage{verbatim}\n")
	out.WriteString("\\usepackage[normalem]{ulem}\n")
	out.WriteString("\\usepackage{hyperref}\n")
	out.WriteString("\n")
	out.WriteString("\\hypersetup{colorlinks,%\n")
	out.WriteString("  citecolor=black,%\n")
	out.WriteString("  filecolor=black,%\n")
	out.WriteString("  linkcolor=black,%\n")
	out.WriteString("  urlcolor=black,%\n")
	out.WriteString("  pdfstartview=FitH,%\n")
	out.WriteString("  breaklinks=true,%\n")
	out.WriteString("  pdfauthor={Blackfriday Markdown Processor v")
	out.WriteString(VERSION)
	out.WriteString("}}\n")
	out.WriteString("\n")
	out.WriteString("\\newcommand{\\HRule}{\\rule{\\linewidth}{0.5mm}}\n")
	out.WriteString("\\addtolength{\\parskip}{0.5\\baselineskip}\n")
	out.WriteString("\\parindent=0pt\n")
	out.WriteString("\n")
	out.WriteString("\\begin{document}\n")
}

func (options *Latex) DocumentFooter(out *bytes.Buffer) {
	out.WriteString("\n\\end{document}\n")
}
