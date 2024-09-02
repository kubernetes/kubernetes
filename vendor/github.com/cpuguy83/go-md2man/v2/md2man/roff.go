package md2man

import (
	"bufio"
	"bytes"
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
	firstDD      bool
	listDepth    int
}

const (
	titleHeader       = ".TH "
	topLevelHeader    = "\n\n.SH "
	secondLevelHdr    = "\n.SH "
	otherHeader       = "\n.SS "
	crTag             = "\n"
	emphTag           = "\\fI"
	emphCloseTag      = "\\fP"
	strongTag         = "\\fB"
	strongCloseTag    = "\\fP"
	breakTag          = "\n.br\n"
	paraTag           = "\n.PP\n"
	hruleTag          = "\n.ti 0\n\\l'\\n(.lu'\n"
	linkTag           = "\n\\[la]"
	linkCloseTag      = "\\[ra]"
	codespanTag       = "\\fB"
	codespanCloseTag  = "\\fR"
	codeTag           = "\n.EX\n"
	codeCloseTag      = ".EE\n" // Do not prepend a newline character since code blocks, by definition, include a newline already (or at least as how blackfriday gives us on).
	quoteTag          = "\n.PP\n.RS\n"
	quoteCloseTag     = "\n.RE\n"
	listTag           = "\n.RS\n"
	listCloseTag      = "\n.RE\n"
	dtTag             = "\n.TP\n"
	dd2Tag            = "\n"
	tableStart        = "\n.TS\nallbox;\n"
	tableEnd          = ".TE\n"
	tableCellStart    = "T{\n"
	tableCellEnd      = "\nT}\n"
	tablePreprocessor = `'\" t`
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
	// We need to walk the tree to check if there are any tables.
	// If there are, we need to enable the roff table preprocessor.
	ast.Walk(func(node *blackfriday.Node, entering bool) blackfriday.WalkStatus {
		if node.Type == blackfriday.Table {
			out(w, tablePreprocessor+"\n")
			return blackfriday.Terminate
		}
		return blackfriday.GoToNext
	})

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
	walkAction := blackfriday.GoToNext

	switch node.Type {
	case blackfriday.Text:
		escapeSpecialChars(w, node.Literal)
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
		// Don't render the link text for automatic links, because this
		// will only duplicate the URL in the roff output.
		// See https://daringfireball.net/projects/markdown/syntax#autolink
		if !bytes.Equal(node.LinkData.Destination, node.FirstChild.Literal) {
			out(w, string(node.FirstChild.Literal))
		}
		// Hyphens in a link must be escaped to avoid word-wrap in the rendered man page.
		escapedLink := strings.ReplaceAll(string(node.LinkData.Destination), "-", "\\-")
		out(w, linkTag+escapedLink+linkCloseTag)
		walkAction = blackfriday.SkipChildren
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
	case blackfriday.TableHead:
	case blackfriday.TableBody:
	case blackfriday.TableRow:
		// no action as cell entries do all the nroff formatting
		return blackfriday.GoToNext
	case blackfriday.TableCell:
		r.handleTableCell(w, node, entering)
	case blackfriday.HTMLSpan:
		// ignore other HTML tags
	case blackfriday.HTMLBlock:
		if bytes.HasPrefix(node.Literal, []byte("<!--")) {
			break // ignore comments, no warning
		}
		fmt.Fprintln(os.Stderr, "WARNING: go-md2man does not handle node type "+node.Type.String())
	default:
		fmt.Fprintln(os.Stderr, "WARNING: go-md2man does not handle node type "+node.Type.String())
	}
	return walkAction
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
		} else if node.ListFlags&blackfriday.ListTypeTerm != 0 {
			// DT (definition term): line just before DD (see below).
			out(w, dtTag)
			r.firstDD = true
		} else if node.ListFlags&blackfriday.ListTypeDefinition != 0 {
			// DD (definition description): line that starts with ": ".
			//
			// We have to distinguish between the first DD and the
			// subsequent ones, as there should be no vertical
			// whitespace between the DT and the first DD.
			if r.firstDD {
				r.firstDD = false
			} else {
				out(w, dd2Tag)
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
		// call walker to count cells (and rows?) so format section can be produced
		columns := countColumns(node)
		out(w, strings.Repeat("l ", columns)+"\n")
		out(w, strings.Repeat("l ", columns)+".\n")
	} else {
		out(w, tableEnd)
	}
}

func (r *roffRenderer) handleTableCell(w io.Writer, node *blackfriday.Node, entering bool) {
	if entering {
		var start string
		if node.Prev != nil && node.Prev.Type == blackfriday.TableCell {
			start = "\t"
		}
		if node.IsHeader {
			start += strongTag
		} else if nodeLiteralSize(node) > 30 {
			start += tableCellStart
		}
		out(w, start)
	} else {
		var end string
		if node.IsHeader {
			end = strongCloseTag
		} else if nodeLiteralSize(node) > 30 {
			end = tableCellEnd
		}
		if node.Next == nil && end != tableCellEnd {
			// Last cell: need to carriage return if we are at the end of the
			// header row and content isn't wrapped in a "tablecell"
			end += crTag
		}
		out(w, end)
	}
}

func nodeLiteralSize(node *blackfriday.Node) int {
	total := 0
	for n := node.FirstChild; n != nil; n = n.FirstChild {
		total += len(n.Literal)
	}
	return total
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

func escapeSpecialChars(w io.Writer, text []byte) {
	scanner := bufio.NewScanner(bytes.NewReader(text))

	// count the number of lines in the text
	// we need to know this to avoid adding a newline after the last line
	n := bytes.Count(text, []byte{'\n'})
	idx := 0

	for scanner.Scan() {
		dt := scanner.Bytes()
		if idx < n {
			idx++
			dt = append(dt, '\n')
		}
		escapeSpecialCharsLine(w, dt)
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}
}

func escapeSpecialCharsLine(w io.Writer, text []byte) {
	for i := 0; i < len(text); i++ {
		// escape initial apostrophe or period
		if len(text) >= 1 && (text[0] == '\'' || text[0] == '.') {
			out(w, "\\&")
		}

		// directly copy normal characters
		org := i

		for i < len(text) && text[i] != '\\' {
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
