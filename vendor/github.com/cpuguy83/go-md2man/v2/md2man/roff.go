package md2man

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/russross/blackfriday/v2"
)

// roffRenderer implements the blackfriday.Renderer interface for creating
// roff format (manpages) from markdown text
type roffRenderer struct {
	extensions   blackfriday.Extensions
	listCounters []int
	firstHeader  bool
	defineTerm   bool
	listDepth    int
}

const (
	titleHeader      = ".TH "
	topLevelHeader   = "\n\n.SH "
	secondLevelHdr   = "\n.SH "
	otherHeader      = "\n.SS "
	crTag            = "\n"
	emphTag          = "\\fI"
	emphCloseTag     = "\\fP"
	strongTag        = "\\fB"
	strongCloseTag   = "\\fP"
	breakTag         = "\n.br\n"
	paraTag          = "\n.PP\n"
	hruleTag         = "\n.ti 0\n\\l'\\n(.lu'\n"
	linkTag          = "\n\\[la]"
	linkCloseTag     = "\\[ra]"
	codespanTag      = "\\fB\\fC"
	codespanCloseTag = "\\fR"
	codeTag          = "\n.PP\n.RS\n\n.nf\n"
	codeCloseTag     = "\n.fi\n.RE\n"
	quoteTag         = "\n.PP\n.RS\n"
	quoteCloseTag    = "\n.RE\n"
	listTag          = "\n.RS\n"
	listCloseTag     = "\n.RE\n"
	arglistTag       = "\n.TP\n"
	tableStart       = "\n.TS\nallbox;\n"
	tableEnd         = ".TE\n"
	tableCellStart   = "T{\n"
	tableCellEnd     = "\nT}\n"
)

// NewRoffRenderer creates a new blackfriday Renderer for generating roff documents
// from markdown
func NewRoffRenderer() *roffRenderer { // nolint: golint
	var extensions blackfriday.Extensions

	extensions |= blackfriday.NoIntraEmphasis
	extensions |= blackfriday.Tables
	extensions |= blackfriday.FencedCode
	extensions |= blackfriday.SpaceHeadings
	extensions |= blackfriday.Footnotes
	extensions |= blackfriday.Titleblock
	extensions |= blackfriday.DefinitionLists
	return &roffRenderer{
		extensions: extensions,
	}
}

// GetExtensions returns the list of extensions used by this renderer implementation
func (r *roffRenderer) GetExtensions() blackfriday.Extensions {
	return r.extensions
}

// RenderHeader handles outputting the header at document start
func (r *roffRenderer) RenderHeader(w io.Writer, ast *blackfriday.Node) {
	// disable hyphenation
	out(w, ".nh\n")
}

// RenderFooter handles outputting the footer at the document end; the roff
// renderer has no footer information
func (r *roffRenderer) RenderFooter(w io.Writer, ast *blackfriday.Node) {
}

// RenderNode is called for each node in a markdown document; based on the node
// type the equivalent roff output is sent to the writer
func (r *roffRenderer) RenderNode(w io.Writer, node *blackfriday.Node, entering bool) blackfriday.WalkStatus {

	var walkAction = blackfriday.GoToNext

	switch node.Type {
	case blackfriday.Text:
		r.handleText(w, node, entering)
	case blackfriday.Softbreak:
		out(w, crTag)
	case blackfriday.Hardbreak:
		out(w, breakTag)
	case blackfriday.Emph:
		if entering {
			out(w, emphTag)
		} else {
			out(w, emphCloseTag)
		}
	case blackfriday.Strong:
		if entering {
			out(w, strongTag)
		} else {
			out(w, strongCloseTag)
		}
	case blackfriday.Link:
		if !entering {
			out(w, linkTag+string(node.LinkData.Destination)+linkCloseTag)
		}
	case blackfriday.Image:
		// ignore images
		walkAction = blackfriday.SkipChildren
	case blackfriday.Code:
		out(w, codespanTag)
		escapeSpecialChars(w, node.Literal)
		out(w, codespanCloseTag)
	case blackfriday.Document:
		break
	case blackfriday.Paragraph:
		// roff .PP markers break lists
		if r.listDepth > 0 {
			return blackfriday.GoToNext
		}
		if entering {
			out(w, paraTag)
		} else {
			out(w, crTag)
		}
	case blackfriday.BlockQuote:
		if entering {
			out(w, quoteTag)
		} else {
			out(w, quoteCloseTag)
		}
	case blackfriday.Heading:
		r.handleHeading(w, node, entering)
	case blackfriday.HorizontalRule:
		out(w, hruleTag)
	case blackfriday.List:
		r.handleList(w, node, entering)
	case blackfriday.Item:
		r.handleItem(w, node, entering)
	case blackfriday.CodeBlock:
		out(w, codeTag)
		escapeSpecialChars(w, node.Literal)
		out(w, codeCloseTag)
	case blackfriday.Table:
		r.handleTable(w, node, entering)
	case blackfriday.TableCell:
		r.handleTableCell(w, node, entering)
	case blackfriday.TableHead:
	case blackfriday.TableBody:
	case blackfriday.TableRow:
		// no action as cell entries do all the nroff formatting
		return blackfriday.GoToNext
	default:
		fmt.Fprintln(os.Stderr, "WARNING: go-md2man does not handle node type "+node.Type.String())
	}
	return walkAction
}

func (r *roffRenderer) handleText(w io.Writer, node *blackfriday.Node, entering bool) {
	var (
		start, end string
	)
	// handle special roff table cell text encapsulation
	if node.Parent.Type == blackfriday.TableCell {
		if len(node.Literal) > 30 {
			start = tableCellStart
			end = tableCellEnd
		} else {
			// end rows that aren't terminated by "tableCellEnd" with a cr if end of row
			if node.Parent.Next == nil && !node.Parent.IsHeader {
				end = crTag
			}
		}
	}
	out(w, start)
	escapeSpecialChars(w, node.Literal)
	out(w, end)
}

func (r *roffRenderer) handleHeading(w io.Writer, node *blackfriday.Node, entering bool) {
	if entering {
		switch node.Level {
		case 1:
			if !r.firstHeader {
				out(w, titleHeader)
				r.firstHeader = true
				break
			}
			out(w, topLevelHeader)
		case 2:
			out(w, secondLevelHdr)
		default:
			out(w, otherHeader)
		}
	}
}

func (r *roffRenderer) handleList(w io.Writer, node *blackfriday.Node, entering bool) {
	openTag := listTag
	closeTag := listCloseTag
	if node.ListFlags&blackfriday.ListTypeDefinition != 0 {
		// tags for definition lists handled within Item node
		openTag = ""
		closeTag = ""
	}
	if entering {
		r.listDepth++
		if node.ListFlags&blackfriday.ListTypeOrdered != 0 {
			r.listCounters = append(r.listCounters, 1)
		}
		out(w, openTag)
	} else {
		if node.ListFlags&blackfriday.ListTypeOrdered != 0 {
			r.listCounters = r.listCounters[:len(r.listCounters)-1]
		}
		out(w, closeTag)
		r.listDepth--
	}
}

func (r *roffRenderer) handleItem(w io.Writer, node *blackfriday.Node, entering bool) {
	if entering {
		if node.ListFlags&blackfriday.ListTypeOrdered != 0 {
			out(w, fmt.Sprintf(".IP \"%3d.\" 5\n", r.listCounters[len(r.listCounters)-1]))
			r.listCounters[len(r.listCounters)-1]++
		} else if node.ListFlags&blackfriday.ListTypeDefinition != 0 {
			// state machine for handling terms and following definitions
			// since blackfriday does not distinguish them properly, nor
			// does it seperate them into separate lists as it should
			if !r.defineTerm {
				out(w, arglistTag)
				r.defineTerm = true
			} else {
				r.defineTerm = false
			}
		} else {
			out(w, ".IP \\(bu 2\n")
		}
	} else {
		out(w, "\n")
	}
}

func (r *roffRenderer) handleTable(w io.Writer, node *blackfriday.Node, entering bool) {
	if entering {
		out(w, tableStart)
		//call walker to count cells (and rows?) so format section can be produced
		columns := countColumns(node)
		out(w, strings.Repeat("l ", columns)+"\n")
		out(w, strings.Repeat("l ", columns)+".\n")
	} else {
		out(w, tableEnd)
	}
}

func (r *roffRenderer) handleTableCell(w io.Writer, node *blackfriday.Node, entering bool) {
	var (
		start, end string
	)
	if node.IsHeader {
		start = codespanTag
		end = codespanCloseTag
	}
	if entering {
		if node.Prev != nil && node.Prev.Type == blackfriday.TableCell {
			out(w, "\t"+start)
		} else {
			out(w, start)
		}
	} else {
		// need to carriage return if we are at the end of the header row
		if node.IsHeader && node.Next == nil {
			end = end + crTag
		}
		out(w, end)
	}
}

// because roff format requires knowing the column count before outputting any table
// data we need to walk a table tree and count the columns
func countColumns(node *blackfriday.Node) int {
	var columns int

	node.Walk(func(node *blackfriday.Node, entering bool) blackfriday.WalkStatus {
		switch node.Type {
		case blackfriday.TableRow:
			if !entering {
				return blackfriday.Terminate
			}
		case blackfriday.TableCell:
			if entering {
				columns++
			}
		default:
		}
		return blackfriday.GoToNext
	})
	return columns
}

func out(w io.Writer, output string) {
	io.WriteString(w, output) // nolint: errcheck
}

func needsBackslash(c byte) bool {
	for _, r := range []byte("-_&\\~") {
		if c == r {
			return true
		}
	}
	return false
}

func escapeSpecialChars(w io.Writer, text []byte) {
	for i := 0; i < len(text); i++ {
		// escape initial apostrophe or period
		if len(text) >= 1 && (text[0] == '\'' || text[0] == '.') {
			out(w, "\\&")
		}

		// directly copy normal characters
		org := i

		for i < len(text) && !needsBackslash(text[i]) {
			i++
		}
		if i > org {
			w.Write(text[org:i]) // nolint: errcheck
		}

		// escape a character
		if i >= len(text) {
			break
		}

		w.Write([]byte{'\\', text[i]}) // nolint: errcheck
	}
}
