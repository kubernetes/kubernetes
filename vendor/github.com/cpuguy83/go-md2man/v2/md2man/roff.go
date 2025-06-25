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
	listCounters []int
	firstHeader  bool
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
	listCloseTag      = ".RE\n"
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
	return &roffRenderer{}
}

// GetExtensions returns the list of extensions used by this renderer implementation
func (*roffRenderer) GetExtensions() blackfriday.Extensions {
	return blackfriday.NoIntraEmphasis |
		blackfriday.Tables |
		blackfriday.FencedCode |
		blackfriday.SpaceHeadings |
		blackfriday.Footnotes |
		blackfriday.Titleblock |
		blackfriday.DefinitionLists
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
		// Special case: format the NAME section as required for proper whatis parsing.
		// Refer to the lexgrog(1) and groff_man(7) manual pages for details.
		if node.Parent != nil &&
			node.Parent.Type == blackfriday.Paragraph &&
			node.Parent.Prev != nil &&
			node.Parent.Prev.Type == blackfriday.Heading &&
			node.Parent.Prev.FirstChild != nil &&
			bytes.EqualFold(node.Parent.Prev.FirstChild.Literal, []byte("NAME")) {
			before, after, found := bytesCut(node.Literal, []byte(" - "))
			escapeSpecialChars(w, before)
			if found {
				out(w, ` \- `)
				escapeSpecialChars(w, after)
			}
		} else {
			escapeSpecialChars(w, node.Literal)
		}
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
		if entering {
			if r.listDepth > 0 {
				// roff .PP markers break lists
				if node.Prev != nil { // continued paragraph
					if node.Prev.Type == blackfriday.List && node.Prev.ListFlags&blackfriday.ListTypeDefinition == 0 {
						out(w, ".IP\n")
					} else {
						out(w, crTag)
					}
				}
			} else if node.Prev != nil && node.Prev.Type == blackfriday.Heading {
				out(w, crTag)
			} else {
				out(w, paraTag)
			}
		} else {
			if node.Next == nil || node.Next.Type != blackfriday.List {
				out(w, crTag)
			}
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
	if (entering && r.listDepth == 0) || (!entering && r.listDepth == 1) {
		openTag = crTag
		closeTag = ""
	}
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
		} else if node.ListFlags&blackfriday.ListTypeDefinition != 0 {
			// DD (definition description): line that starts with ": ".
			//
			// We have to distinguish between the first DD and the
			// subsequent ones, as there should be no vertical
			// whitespace between the DT and the first DD.
			if node.Prev != nil && node.Prev.ListFlags&(blackfriday.ListTypeTerm|blackfriday.ListTypeDefinition) == blackfriday.ListTypeDefinition {
				if node.Prev.Type == blackfriday.Item &&
					node.Prev.LastChild != nil &&
					node.Prev.LastChild.Type == blackfriday.List &&
					node.Prev.LastChild.ListFlags&blackfriday.ListTypeDefinition == 0 {
					out(w, ".IP\n")
				} else {
					out(w, dd2Tag)
				}
			}
		} else {
			out(w, ".IP \\(bu 2\n")
		}
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

// bytesCut is a copy of [bytes.Cut] to provide compatibility with go1.17
// and older. We can remove this once we drop support  for go1.17 and older.
func bytesCut(s, sep []byte) (before, after []byte, found bool) {
	if i := bytes.Index(s, sep); i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, nil, false
}
