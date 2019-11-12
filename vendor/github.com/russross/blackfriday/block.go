//
// Blackfriday Markdown Processor
// Available at http://github.com/russross/blackfriday
//
// Copyright © 2011 Russ Ross <russ@russross.com>.
// Distributed under the Simplified BSD License.
// See README.md for details.
//

//
// Functions to parse block-level elements.
//

package blackfriday

import (
	"bytes"
	"html"
	"regexp"

	"github.com/shurcooL/sanitized_anchor_name"
)

const (
	charEntity = "&(?:#x[a-f0-9]{1,8}|#[0-9]{1,8}|[a-z][a-z0-9]{1,31});"
	escapable  = "[!\"#$%&'()*+,./:;<=>?@[\\\\\\]^_`{|}~-]"
)

var (
	reBackslashOrAmp      = regexp.MustCompile("[\\&]")
	reEntityOrEscapedChar = regexp.MustCompile("(?i)\\\\" + escapable + "|" + charEntity)
)

// Parse block-level data.
// Note: this function and many that it calls assume that
// the input buffer ends with a newline.
func (p *Markdown) block(data []byte) {
	// this is called recursively: enforce a maximum depth
	if p.nesting >= p.maxNesting {
		return
	}
	p.nesting++

	// parse out one block-level construct at a time
	for len(data) > 0 {
		// prefixed heading:
		//
		// # Heading 1
		// ## Heading 2
		// ...
		// ###### Heading 6
		if p.isPrefixHeading(data) {
			data = data[p.prefixHeading(data):]
			continue
		}

		// block of preformatted HTML:
		//
		// <div>
		//     ...
		// </div>
		if data[0] == '<' {
			if i := p.html(data, true); i > 0 {
				data = data[i:]
				continue
			}
		}

		// title block
		//
		// % stuff
		// % more stuff
		// % even more stuff
		if p.extensions&Titleblock != 0 {
			if data[0] == '%' {
				if i := p.titleBlock(data, true); i > 0 {
					data = data[i:]
					continue
				}
			}
		}

		// blank lines.  note: returns the # of bytes to skip
		if i := p.isEmpty(data); i > 0 {
			data = data[i:]
			continue
		}

		// indented code block:
		//
		//     func max(a, b int) int {
		//         if a > b {
		//             return a
		//         }
		//         return b
		//      }
		if p.codePrefix(data) > 0 {
			data = data[p.code(data):]
			continue
		}

		// fenced code block:
		//
		// ``` go
		// func fact(n int) int {
		//     if n <= 1 {
		//         return n
		//     }
		//     return n * fact(n-1)
		// }
		// ```
		if p.extensions&FencedCode != 0 {
			if i := p.fencedCodeBlock(data, true); i > 0 {
				data = data[i:]
				continue
			}
		}

		// horizontal rule:
		//
		// ------
		// or
		// ******
		// or
		// ______
		if p.isHRule(data) {
			p.addBlock(HorizontalRule, nil)
			var i int
			for i = 0; i < len(data) && data[i] != '\n'; i++ {
			}
			data = data[i:]
			continue
		}

		// block quote:
		//
		// > A big quote I found somewhere
		// > on the web
		if p.quotePrefix(data) > 0 {
			data = data[p.quote(data):]
			continue
		}

		// table:
		//
		// Name  | Age | Phone
		// ------|-----|---------
		// Bob   | 31  | 555-1234
		// Alice | 27  | 555-4321
		if p.extensions&Tables != 0 {
			if i := p.table(data); i > 0 {
				data = data[i:]
				continue
			}
		}

		// an itemized/unordered list:
		//
		// * Item 1
		// * Item 2
		//
		// also works with + or -
		if p.uliPrefix(data) > 0 {
			data = data[p.list(data, 0):]
			continue
		}

		// a numbered/ordered list:
		//
		// 1. Item 1
		// 2. Item 2
		if p.oliPrefix(data) > 0 {
			data = data[p.list(data, ListTypeOrdered):]
			continue
		}

		// definition lists:
		//
		// Term 1
		// :   Definition a
		// :   Definition b
		//
		// Term 2
		// :   Definition c
		if p.extensions&DefinitionLists != 0 {
			if p.dliPrefix(data) > 0 {
				data = data[p.list(data, ListTypeDefinition):]
				continue
			}
		}

		// anything else must look like a normal paragraph
		// note: this finds underlined headings, too
		data = data[p.paragraph(data):]
	}

	p.nesting--
}

func (p *Markdown) addBlock(typ NodeType, content []byte) *Node {
	p.closeUnmatchedBlocks()
	container := p.addChild(typ, 0)
	container.content = content
	return container
}

func (p *Markdown) isPrefixHeading(data []byte) bool {
	if data[0] != '#' {
		return false
	}

	if p.extensions&SpaceHeadings != 0 {
		level := 0
		for level < 6 && level < len(data) && data[level] == '#' {
			level++
		}
		if level == len(data) || data[level] != ' ' {
			return false
		}
	}
	return true
}

func (p *Markdown) prefixHeading(data []byte) int {
	level := 0
	for level < 6 && level < len(data) && data[level] == '#' {
		level++
	}
	i := skipChar(data, level, ' ')
	end := skipUntilChar(data, i, '\n')
	skip := end
	id := ""
	if p.extensions&HeadingIDs != 0 {
		j, k := 0, 0
		// find start/end of heading id
		for j = i; j < end-1 && (data[j] != '{' || data[j+1] != '#'); j++ {
		}
		for k = j + 1; k < end && data[k] != '}'; k++ {
		}
		// extract heading id iff found
		if j < end && k < end {
			id = string(data[j+2 : k])
			end = j
			skip = k + 1
			for end > 0 && data[end-1] == ' ' {
				end--
			}
		}
	}
	for end > 0 && data[end-1] == '#' {
		if isBackslashEscaped(data, end-1) {
			break
		}
		end--
	}
	for end > 0 && data[end-1] == ' ' {
		end--
	}
	if end > i {
		if id == "" && p.extensions&AutoHeadingIDs != 0 {
			id = sanitized_anchor_name.Create(string(data[i:end]))
		}
		block := p.addBlock(Heading, data[i:end])
		block.HeadingID = id
		block.Level = level
	}
	return skip
}

func (p *Markdown) isUnderlinedHeading(data []byte) int {
	// test of level 1 heading
	if data[0] == '=' {
		i := skipChar(data, 1, '=')
		i = skipChar(data, i, ' ')
		if i < len(data) && data[i] == '\n' {
			return 1
		}
		return 0
	}

	// test of level 2 heading
	if data[0] == '-' {
		i := skipChar(data, 1, '-')
		i = skipChar(data, i, ' ')
		if i < len(data) && data[i] == '\n' {
			return 2
		}
		return 0
	}

	return 0
}

func (p *Markdown) titleBlock(data []byte, doRender bool) int {
	if data[0] != '%' {
		return 0
	}
	splitData := bytes.Split(data, []byte("\n"))
	var i int
	for idx, b := range splitData {
		if !bytes.HasPrefix(b, []byte("%")) {
			i = idx // - 1
			break
		}
	}

	data = bytes.Join(splitData[0:i], []byte("\n"))
	consumed := len(data)
	data = bytes.TrimPrefix(data, []byte("% "))
	data = bytes.Replace(data, []byte("\n% "), []byte("\n"), -1)
	block := p.addBlock(Heading, data)
	block.Level = 1
	block.IsTitleblock = true

	return consumed
}

func (p *Markdown) html(data []byte, doRender bool) int {
	var i, j int

	// identify the opening tag
	if data[0] != '<' {
		return 0
	}
	curtag, tagfound := p.htmlFindTag(data[1:])

	// handle special cases
	if !tagfound {
		// check for an HTML comment
		if size := p.htmlComment(data, doRender); size > 0 {
			return size
		}

		// check for an <hr> tag
		if size := p.htmlHr(data, doRender); size > 0 {
			return size
		}

		// no special case recognized
		return 0
	}

	// look for an unindented matching closing tag
	// followed by a blank line
	found := false
	/*
		closetag := []byte("\n</" + curtag + ">")
		j = len(curtag) + 1
		for !found {
			// scan for a closing tag at the beginning of a line
			if skip := bytes.Index(data[j:], closetag); skip >= 0 {
				j += skip + len(closetag)
			} else {
				break
			}

			// see if it is the only thing on the line
			if skip := p.isEmpty(data[j:]); skip > 0 {
				// see if it is followed by a blank line/eof
				j += skip
				if j >= len(data) {
					found = true
					i = j
				} else {
					if skip := p.isEmpty(data[j:]); skip > 0 {
						j += skip
						found = true
						i = j
					}
				}
			}
		}
	*/

	// if not found, try a second pass looking for indented match
	// but not if tag is "ins" or "del" (following original Markdown.pl)
	if !found && curtag != "ins" && curtag != "del" {
		i = 1
		for i < len(data) {
			i++
			for i < len(data) && !(data[i-1] == '<' && data[i] == '/') {
				i++
			}

			if i+2+len(curtag) >= len(data) {
				break
			}

			j = p.htmlFindEnd(curtag, data[i-1:])

			if j > 0 {
				i += j - 1
				found = true
				break
			}
		}
	}

	if !found {
		return 0
	}

	// the end of the block has been found
	if doRender {
		// trim newlines
		end := i
		for end > 0 && data[end-1] == '\n' {
			end--
		}
		finalizeHTMLBlock(p.addBlock(HTMLBlock, data[:end]))
	}

	return i
}

func finalizeHTMLBlock(block *Node) {
	block.Literal = block.content
	block.content = nil
}

// HTML comment, lax form
func (p *Markdown) htmlComment(data []byte, doRender bool) int {
	i := p.inlineHTMLComment(data)
	// needs to end with a blank line
	if j := p.isEmpty(data[i:]); j > 0 {
		size := i + j
		if doRender {
			// trim trailing newlines
			end := size
			for end > 0 && data[end-1] == '\n' {
				end--
			}
			block := p.addBlock(HTMLBlock, data[:end])
			finalizeHTMLBlock(block)
		}
		return size
	}
	return 0
}

// HR, which is the only self-closing block tag considered
func (p *Markdown) htmlHr(data []byte, doRender bool) int {
	if len(data) < 4 {
		return 0
	}
	if data[0] != '<' || (data[1] != 'h' && data[1] != 'H') || (data[2] != 'r' && data[2] != 'R') {
		return 0
	}
	if data[3] != ' ' && data[3] != '/' && data[3] != '>' {
		// not an <hr> tag after all; at least not a valid one
		return 0
	}
	i := 3
	for i < len(data) && data[i] != '>' && data[i] != '\n' {
		i++
	}
	if i < len(data) && data[i] == '>' {
		i++
		if j := p.isEmpty(data[i:]); j > 0 {
			size := i + j
			if doRender {
				// trim newlines
				end := size
				for end > 0 && data[end-1] == '\n' {
					end--
				}
				finalizeHTMLBlock(p.addBlock(HTMLBlock, data[:end]))
			}
			return size
		}
	}
	return 0
}

func (p *Markdown) htmlFindTag(data []byte) (string, bool) {
	i := 0
	for i < len(data) && isalnum(data[i]) {
		i++
	}
	key := string(data[:i])
	if _, ok := blockTags[key]; ok {
		return key, true
	}
	return "", false
}

func (p *Markdown) htmlFindEnd(tag string, data []byte) int {
	// assume data[0] == '<' && data[1] == '/' already tested
	if tag == "hr" {
		return 2
	}
	// check if tag is a match
	closetag := []byte("</" + tag + ">")
	if !bytes.HasPrefix(data, closetag) {
		return 0
	}
	i := len(closetag)

	// check that the rest of the line is blank
	skip := 0
	if skip = p.isEmpty(data[i:]); skip == 0 {
		return 0
	}
	i += skip
	skip = 0

	if i >= len(data) {
		return i
	}

	if p.extensions&LaxHTMLBlocks != 0 {
		return i
	}
	if skip = p.isEmpty(data[i:]); skip == 0 {
		// following line must be blank
		return 0
	}

	return i + skip
}

func (*Markdown) isEmpty(data []byte) int {
	// it is okay to call isEmpty on an empty buffer
	if len(data) == 0 {
		return 0
	}

	var i int
	for i = 0; i < len(data) && data[i] != '\n'; i++ {
		if data[i] != ' ' && data[i] != '\t' {
			return 0
		}
	}
	if i < len(data) && data[i] == '\n' {
		i++
	}
	return i
}

func (*Markdown) isHRule(data []byte) bool {
	i := 0

	// skip up to three spaces
	for i < 3 && data[i] == ' ' {
		i++
	}

	// look at the hrule char
	if data[i] != '*' && data[i] != '-' && data[i] != '_' {
		return false
	}
	c := data[i]

	// the whole line must be the char or whitespace
	n := 0
	for i < len(data) && data[i] != '\n' {
		switch {
		case data[i] == c:
			n++
		case data[i] != ' ':
			return false
		}
		i++
	}

	return n >= 3
}

// isFenceLine checks if there's a fence line (e.g., ``` or ``` go) at the beginning of data,
// and returns the end index if so, or 0 otherwise. It also returns the marker found.
// If syntax is not nil, it gets set to the syntax specified in the fence line.
func isFenceLine(data []byte, syntax *string, oldmarker string) (end int, marker string) {
	i, size := 0, 0

	// skip up to three spaces
	for i < len(data) && i < 3 && data[i] == ' ' {
		i++
	}

	// check for the marker characters: ~ or `
	if i >= len(data) {
		return 0, ""
	}
	if data[i] != '~' && data[i] != '`' {
		return 0, ""
	}

	c := data[i]

	// the whole line must be the same char or whitespace
	for i < len(data) && data[i] == c {
		size++
		i++
	}

	// the marker char must occur at least 3 times
	if size < 3 {
		return 0, ""
	}
	marker = string(data[i-size : i])

	// if this is the end marker, it must match the beginning marker
	if oldmarker != "" && marker != oldmarker {
		return 0, ""
	}

	// TODO(shurcooL): It's probably a good idea to simplify the 2 code paths here
	// into one, always get the syntax, and discard it if the caller doesn't care.
	if syntax != nil {
		syn := 0
		i = skipChar(data, i, ' ')

		if i >= len(data) {
			if i == len(data) {
				return i, marker
			}
			return 0, ""
		}

		syntaxStart := i

		if data[i] == '{' {
			i++
			syntaxStart++

			for i < len(data) && data[i] != '}' && data[i] != '\n' {
				syn++
				i++
			}

			if i >= len(data) || data[i] != '}' {
				return 0, ""
			}

			// strip all whitespace at the beginning and the end
			// of the {} block
			for syn > 0 && isspace(data[syntaxStart]) {
				syntaxStart++
				syn--
			}

			for syn > 0 && isspace(data[syntaxStart+syn-1]) {
				syn--
			}

			i++
		} else {
			for i < len(data) && !isspace(data[i]) {
				syn++
				i++
			}
		}

		*syntax = string(data[syntaxStart : syntaxStart+syn])
	}

	i = skipChar(data, i, ' ')
	if i >= len(data) || data[i] != '\n' {
		if i == len(data) {
			return i, marker
		}
		return 0, ""
	}
	return i + 1, marker // Take newline into account.
}

// fencedCodeBlock returns the end index if data contains a fenced code block at the beginning,
// or 0 otherwise. It writes to out if doRender is true, otherwise it has no side effects.
// If doRender is true, a final newline is mandatory to recognize the fenced code block.
func (p *Markdown) fencedCodeBlock(data []byte, doRender bool) int {
	var syntax string
	beg, marker := isFenceLine(data, &syntax, "")
	if beg == 0 || beg >= len(data) {
		return 0
	}

	var work bytes.Buffer
	work.Write([]byte(syntax))
	work.WriteByte('\n')

	for {
		// safe to assume beg < len(data)

		// check for the end of the code block
		fenceEnd, _ := isFenceLine(data[beg:], nil, marker)
		if fenceEnd != 0 {
			beg += fenceEnd
			break
		}

		// copy the current line
		end := skipUntilChar(data, beg, '\n') + 1

		// did we reach the end of the buffer without a closing marker?
		if end >= len(data) {
			return 0
		}

		// verbatim copy to the working buffer
		if doRender {
			work.Write(data[beg:end])
		}
		beg = end
	}

	if doRender {
		block := p.addBlock(CodeBlock, work.Bytes()) // TODO: get rid of temp buffer
		block.IsFenced = true
		finalizeCodeBlock(block)
	}

	return beg
}

func unescapeChar(str []byte) []byte {
	if str[0] == '\\' {
		return []byte{str[1]}
	}
	return []byte(html.UnescapeString(string(str)))
}

func unescapeString(str []byte) []byte {
	if reBackslashOrAmp.Match(str) {
		return reEntityOrEscapedChar.ReplaceAllFunc(str, unescapeChar)
	}
	return str
}

func finalizeCodeBlock(block *Node) {
	if block.IsFenced {
		newlinePos := bytes.IndexByte(block.content, '\n')
		firstLine := block.content[:newlinePos]
		rest := block.content[newlinePos+1:]
		block.Info = unescapeString(bytes.Trim(firstLine, "\n"))
		block.Literal = rest
	} else {
		block.Literal = block.content
	}
	block.content = nil
}

func (p *Markdown) table(data []byte) int {
	table := p.addBlock(Table, nil)
	i, columns := p.tableHeader(data)
	if i == 0 {
		p.tip = table.Parent
		table.Unlink()
		return 0
	}

	p.addBlock(TableBody, nil)

	for i < len(data) {
		pipes, rowStart := 0, i
		for ; i < len(data) && data[i] != '\n'; i++ {
			if data[i] == '|' {
				pipes++
			}
		}

		if pipes == 0 {
			i = rowStart
			break
		}

		// include the newline in data sent to tableRow
		if i < len(data) && data[i] == '\n' {
			i++
		}
		p.tableRow(data[rowStart:i], columns, false)
	}

	return i
}

// check if the specified position is preceded by an odd number of backslashes
func isBackslashEscaped(data []byte, i int) bool {
	backslashes := 0
	for i-backslashes-1 >= 0 && data[i-backslashes-1] == '\\' {
		backslashes++
	}
	return backslashes&1 == 1
}

func (p *Markdown) tableHeader(data []byte) (size int, columns []CellAlignFlags) {
	i := 0
	colCount := 1
	for i = 0; i < len(data) && data[i] != '\n'; i++ {
		if data[i] == '|' && !isBackslashEscaped(data, i) {
			colCount++
		}
	}

	// doesn't look like a table header
	if colCount == 1 {
		return
	}

	// include the newline in the data sent to tableRow
	j := i
	if j < len(data) && data[j] == '\n' {
		j++
	}
	header := data[:j]

	// column count ignores pipes at beginning or end of line
	if data[0] == '|' {
		colCount--
	}
	if i > 2 && data[i-1] == '|' && !isBackslashEscaped(data, i-1) {
		colCount--
	}

	columns = make([]CellAlignFlags, colCount)

	// move on to the header underline
	i++
	if i >= len(data) {
		return
	}

	if data[i] == '|' && !isBackslashEscaped(data, i) {
		i++
	}
	i = skipChar(data, i, ' ')

	// each column header is of form: / *:?-+:? *|/ with # dashes + # colons >= 3
	// and trailing | optional on last column
	col := 0
	for i < len(data) && data[i] != '\n' {
		dashes := 0

		if data[i] == ':' {
			i++
			columns[col] |= TableAlignmentLeft
			dashes++
		}
		for i < len(data) && data[i] == '-' {
			i++
			dashes++
		}
		if i < len(data) && data[i] == ':' {
			i++
			columns[col] |= TableAlignmentRight
			dashes++
		}
		for i < len(data) && data[i] == ' ' {
			i++
		}
		if i == len(data) {
			return
		}
		// end of column test is messy
		switch {
		case dashes < 3:
			// not a valid column
			return

		case data[i] == '|' && !isBackslashEscaped(data, i):
			// marker found, now skip past trailing whitespace
			col++
			i++
			for i < len(data) && data[i] == ' ' {
				i++
			}

			// trailing junk found after last column
			if col >= colCount && i < len(data) && data[i] != '\n' {
				return
			}

		case (data[i] != '|' || isBackslashEscaped(data, i)) && col+1 < colCount:
			// something else found where marker was required
			return

		case data[i] == '\n':
			// marker is optional for the last column
			col++

		default:
			// trailing junk found after last column
			return
		}
	}
	if col != colCount {
		return
	}

	p.addBlock(TableHead, nil)
	p.tableRow(header, columns, true)
	size = i
	if size < len(data) && data[size] == '\n' {
		size++
	}
	return
}

func (p *Markdown) tableRow(data []byte, columns []CellAlignFlags, header bool) {
	p.addBlock(TableRow, nil)
	i, col := 0, 0

	if data[i] == '|' && !isBackslashEscaped(data, i) {
		i++
	}

	for col = 0; col < len(columns) && i < len(data); col++ {
		for i < len(data) && data[i] == ' ' {
			i++
		}

		cellStart := i

		for i < len(data) && (data[i] != '|' || isBackslashEscaped(data, i)) && data[i] != '\n' {
			i++
		}

		cellEnd := i

		// skip the end-of-cell marker, possibly taking us past end of buffer
		i++

		for cellEnd > cellStart && cellEnd-1 < len(data) && data[cellEnd-1] == ' ' {
			cellEnd--
		}

		cell := p.addBlock(TableCell, data[cellStart:cellEnd])
		cell.IsHeader = header
		cell.Align = columns[col]
	}

	// pad it out with empty columns to get the right number
	for ; col < len(columns); col++ {
		cell := p.addBlock(TableCell, nil)
		cell.IsHeader = header
		cell.Align = columns[col]
	}

	// silently ignore rows with too many cells
}

// returns blockquote prefix length
func (p *Markdown) quotePrefix(data []byte) int {
	i := 0
	for i < 3 && i < len(data) && data[i] == ' ' {
		i++
	}
	if i < len(data) && data[i] == '>' {
		if i+1 < len(data) && data[i+1] == ' ' {
			return i + 2
		}
		return i + 1
	}
	return 0
}

// blockquote ends with at least one blank line
// followed by something without a blockquote prefix
func (p *Markdown) terminateBlockquote(data []byte, beg, end int) bool {
	if p.isEmpty(data[beg:]) <= 0 {
		return false
	}
	if end >= len(data) {
		return true
	}
	return p.quotePrefix(data[end:]) == 0 && p.isEmpty(data[end:]) == 0
}

// parse a blockquote fragment
func (p *Markdown) quote(data []byte) int {
	block := p.addBlock(BlockQuote, nil)
	var raw bytes.Buffer
	beg, end := 0, 0
	for beg < len(data) {
		end = beg
		// Step over whole lines, collecting them. While doing that, check for
		// fenced code and if one's found, incorporate it altogether,
		// irregardless of any contents inside it
		for end < len(data) && data[end] != '\n' {
			if p.extensions&FencedCode != 0 {
				if i := p.fencedCodeBlock(data[end:], false); i > 0 {
					// -1 to compensate for the extra end++ after the loop:
					end += i - 1
					break
				}
			}
			end++
		}
		if end < len(data) && data[end] == '\n' {
			end++
		}
		if pre := p.quotePrefix(data[beg:]); pre > 0 {
			// skip the prefix
			beg += pre
		} else if p.terminateBlockquote(data, beg, end) {
			break
		}
		// this line is part of the blockquote
		raw.Write(data[beg:end])
		beg = end
	}
	p.block(raw.Bytes())
	p.finalize(block)
	return end
}

// returns prefix length for block code
func (p *Markdown) codePrefix(data []byte) int {
	if len(data) >= 1 && data[0] == '\t' {
		return 1
	}
	if len(data) >= 4 && data[0] == ' ' && data[1] == ' ' && data[2] == ' ' && data[3] == ' ' {
		return 4
	}
	return 0
}

func (p *Markdown) code(data []byte) int {
	var work bytes.Buffer

	i := 0
	for i < len(data) {
		beg := i
		for i < len(data) && data[i] != '\n' {
			i++
		}
		if i < len(data) && data[i] == '\n' {
			i++
		}

		blankline := p.isEmpty(data[beg:i]) > 0
		if pre := p.codePrefix(data[beg:i]); pre > 0 {
			beg += pre
		} else if !blankline {
			// non-empty, non-prefixed line breaks the pre
			i = beg
			break
		}

		// verbatim copy to the working buffer
		if blankline {
			work.WriteByte('\n')
		} else {
			work.Write(data[beg:i])
		}
	}

	// trim all the \n off the end of work
	workbytes := work.Bytes()
	eol := len(workbytes)
	for eol > 0 && workbytes[eol-1] == '\n' {
		eol--
	}
	if eol != len(workbytes) {
		work.Truncate(eol)
	}

	work.WriteByte('\n')

	block := p.addBlock(CodeBlock, work.Bytes()) // TODO: get rid of temp buffer
	block.IsFenced = false
	finalizeCodeBlock(block)

	return i
}

// returns unordered list item prefix
func (p *Markdown) uliPrefix(data []byte) int {
	i := 0
	// start with up to 3 spaces
	for i < len(data) && i < 3 && data[i] == ' ' {
		i++
	}
	if i >= len(data)-1 {
		return 0
	}
	// need one of {'*', '+', '-'} followed by a space or a tab
	if (data[i] != '*' && data[i] != '+' && data[i] != '-') ||
		(data[i+1] != ' ' && data[i+1] != '\t') {
		return 0
	}
	return i + 2
}

// returns ordered list item prefix
func (p *Markdown) oliPrefix(data []byte) int {
	i := 0

	// start with up to 3 spaces
	for i < 3 && i < len(data) && data[i] == ' ' {
		i++
	}

	// count the digits
	start := i
	for i < len(data) && data[i] >= '0' && data[i] <= '9' {
		i++
	}
	if start == i || i >= len(data)-1 {
		return 0
	}

	// we need >= 1 digits followed by a dot and a space or a tab
	if data[i] != '.' || !(data[i+1] == ' ' || data[i+1] == '\t') {
		return 0
	}
	return i + 2
}

// returns definition list item prefix
func (p *Markdown) dliPrefix(data []byte) int {
	if len(data) < 2 {
		return 0
	}
	i := 0
	// need a ':' followed by a space or a tab
	if data[i] != ':' || !(data[i+1] == ' ' || data[i+1] == '\t') {
		return 0
	}
	for i < len(data) && data[i] == ' ' {
		i++
	}
	return i + 2
}

// parse ordered or unordered list block
func (p *Markdown) list(data []byte, flags ListType) int {
	i := 0
	flags |= ListItemBeginningOfList
	block := p.addBlock(List, nil)
	block.ListFlags = flags
	block.Tight = true

	for i < len(data) {
		skip := p.listItem(data[i:], &flags)
		if flags&ListItemContainsBlock != 0 {
			block.ListData.Tight = false
		}
		i += skip
		if skip == 0 || flags&ListItemEndOfList != 0 {
			break
		}
		flags &= ^ListItemBeginningOfList
	}

	above := block.Parent
	finalizeList(block)
	p.tip = above
	return i
}

// Returns true if block ends with a blank line, descending if needed
// into lists and sublists.
func endsWithBlankLine(block *Node) bool {
	// TODO: figure this out. Always false now.
	for block != nil {
		//if block.lastLineBlank {
		//return true
		//}
		t := block.Type
		if t == List || t == Item {
			block = block.LastChild
		} else {
			break
		}
	}
	return false
}

func finalizeList(block *Node) {
	block.open = false
	item := block.FirstChild
	for item != nil {
		// check for non-final list item ending with blank line:
		if endsWithBlankLine(item) && item.Next != nil {
			block.ListData.Tight = false
			break
		}
		// recurse into children of list item, to see if there are spaces
		// between any of them:
		subItem := item.FirstChild
		for subItem != nil {
			if endsWithBlankLine(subItem) && (item.Next != nil || subItem.Next != nil) {
				block.ListData.Tight = false
				break
			}
			subItem = subItem.Next
		}
		item = item.Next
	}
}

// Parse a single list item.
// Assumes initial prefix is already removed if this is a sublist.
func (p *Markdown) listItem(data []byte, flags *ListType) int {
	// keep track of the indentation of the first line
	itemIndent := 0
	if data[0] == '\t' {
		itemIndent += 4
	} else {
		for itemIndent < 3 && data[itemIndent] == ' ' {
			itemIndent++
		}
	}

	var bulletChar byte = '*'
	i := p.uliPrefix(data)
	if i == 0 {
		i = p.oliPrefix(data)
	} else {
		bulletChar = data[i-2]
	}
	if i == 0 {
		i = p.dliPrefix(data)
		// reset definition term flag
		if i > 0 {
			*flags &= ^ListTypeTerm
		}
	}
	if i == 0 {
		// if in definition list, set term flag and continue
		if *flags&ListTypeDefinition != 0 {
			*flags |= ListTypeTerm
		} else {
			return 0
		}
	}

	// skip leading whitespace on first line
	for i < len(data) && data[i] == ' ' {
		i++
	}

	// find the end of the line
	line := i
	for i > 0 && i < len(data) && data[i-1] != '\n' {
		i++
	}

	// get working buffer
	var raw bytes.Buffer

	// put the first line into the working buffer
	raw.Write(data[line:i])
	line = i

	// process the following lines
	containsBlankLine := false
	sublist := 0

gatherlines:
	for line < len(data) {
		i++

		// find the end of this line
		for i < len(data) && data[i-1] != '\n' {
			i++
		}

		// if it is an empty line, guess that it is part of this item
		// and move on to the next line
		if p.isEmpty(data[line:i]) > 0 {
			containsBlankLine = true
			line = i
			continue
		}

		// calculate the indentation
		indent := 0
		indentIndex := 0
		if data[line] == '\t' {
			indentIndex++
			indent += 4
		} else {
			for indent < 4 && line+indent < i && data[line+indent] == ' ' {
				indent++
				indentIndex++
			}
		}

		chunk := data[line+indentIndex : i]

		// evaluate how this line fits in
		switch {
		// is this a nested list item?
		case (p.uliPrefix(chunk) > 0 && !p.isHRule(chunk)) ||
			p.oliPrefix(chunk) > 0 ||
			p.dliPrefix(chunk) > 0:

			if containsBlankLine {
				*flags |= ListItemContainsBlock
			}

			// to be a nested list, it must be indented more
			// if not, it is the next item in the same list
			if indent <= itemIndent {
				break gatherlines
			}

			// is this the first item in the nested list?
			if sublist == 0 {
				sublist = raw.Len()
			}

		// is this a nested prefix heading?
		case p.isPrefixHeading(chunk):
			// if the heading is not indented, it is not nested in the list
			// and thus ends the list
			if containsBlankLine && indent < 4 {
				*flags |= ListItemEndOfList
				break gatherlines
			}
			*flags |= ListItemContainsBlock

		// anything following an empty line is only part
		// of this item if it is indented 4 spaces
		// (regardless of the indentation of the beginning of the item)
		case containsBlankLine && indent < 4:
			if *flags&ListTypeDefinition != 0 && i < len(data)-1 {
				// is the next item still a part of this list?
				next := i
				for next < len(data) && data[next] != '\n' {
					next++
				}
				for next < len(data)-1 && data[next] == '\n' {
					next++
				}
				if i < len(data)-1 && data[i] != ':' && data[next] != ':' {
					*flags |= ListItemEndOfList
				}
			} else {
				*flags |= ListItemEndOfList
			}
			break gatherlines

		// a blank line means this should be parsed as a block
		case containsBlankLine:
			raw.WriteByte('\n')
			*flags |= ListItemContainsBlock
		}

		// if this line was preceded by one or more blanks,
		// re-introduce the blank into the buffer
		if containsBlankLine {
			containsBlankLine = false
			raw.WriteByte('\n')
		}

		// add the line into the working buffer without prefix
		raw.Write(data[line+indentIndex : i])

		line = i
	}

	rawBytes := raw.Bytes()

	block := p.addBlock(Item, nil)
	block.ListFlags = *flags
	block.Tight = false
	block.BulletChar = bulletChar
	block.Delimiter = '.' // Only '.' is possible in Markdown, but ')' will also be possible in CommonMark

	// render the contents of the list item
	if *flags&ListItemContainsBlock != 0 && *flags&ListTypeTerm == 0 {
		// intermediate render of block item, except for definition term
		if sublist > 0 {
			p.block(rawBytes[:sublist])
			p.block(rawBytes[sublist:])
		} else {
			p.block(rawBytes)
		}
	} else {
		// intermediate render of inline item
		if sublist > 0 {
			child := p.addChild(Paragraph, 0)
			child.content = rawBytes[:sublist]
			p.block(rawBytes[sublist:])
		} else {
			child := p.addChild(Paragraph, 0)
			child.content = rawBytes
		}
	}
	return line
}

// render a single paragraph that has already been parsed out
func (p *Markdown) renderParagraph(data []byte) {
	if len(data) == 0 {
		return
	}

	// trim leading spaces
	beg := 0
	for data[beg] == ' ' {
		beg++
	}

	end := len(data)
	// trim trailing newline
	if data[len(data)-1] == '\n' {
		end--
	}

	// trim trailing spaces
	for end > beg && data[end-1] == ' ' {
		end--
	}

	p.addBlock(Paragraph, data[beg:end])
}

func (p *Markdown) paragraph(data []byte) int {
	// prev: index of 1st char of previous line
	// line: index of 1st char of current line
	// i: index of cursor/end of current line
	var prev, line, i int
	tabSize := TabSizeDefault
	if p.extensions&TabSizeEight != 0 {
		tabSize = TabSizeDouble
	}
	// keep going until we find something to mark the end of the paragraph
	for i < len(data) {
		// mark the beginning of the current line
		prev = line
		current := data[i:]
		line = i

		// did we find a reference or a footnote? If so, end a paragraph
		// preceding it and report that we have consumed up to the end of that
		// reference:
		if refEnd := isReference(p, current, tabSize); refEnd > 0 {
			p.renderParagraph(data[:i])
			return i + refEnd
		}

		// did we find a blank line marking the end of the paragraph?
		if n := p.isEmpty(current); n > 0 {
			// did this blank line followed by a definition list item?
			if p.extensions&DefinitionLists != 0 {
				if i < len(data)-1 && data[i+1] == ':' {
					return p.list(data[prev:], ListTypeDefinition)
				}
			}

			p.renderParagraph(data[:i])
			return i + n
		}

		// an underline under some text marks a heading, so our paragraph ended on prev line
		if i > 0 {
			if level := p.isUnderlinedHeading(current); level > 0 {
				// render the paragraph
				p.renderParagraph(data[:prev])

				// ignore leading and trailing whitespace
				eol := i - 1
				for prev < eol && data[prev] == ' ' {
					prev++
				}
				for eol > prev && data[eol-1] == ' ' {
					eol--
				}

				id := ""
				if p.extensions&AutoHeadingIDs != 0 {
					id = sanitized_anchor_name.Create(string(data[prev:eol]))
				}

				block := p.addBlock(Heading, data[prev:eol])
				block.Level = level
				block.HeadingID = id

				// find the end of the underline
				for i < len(data) && data[i] != '\n' {
					i++
				}
				return i
			}
		}

		// if the next line starts a block of HTML, then the paragraph ends here
		if p.extensions&LaxHTMLBlocks != 0 {
			if data[i] == '<' && p.html(current, false) > 0 {
				// rewind to before the HTML block
				p.renderParagraph(data[:i])
				return i
			}
		}

		// if there's a prefixed heading or a horizontal rule after this, paragraph is over
		if p.isPrefixHeading(current) || p.isHRule(current) {
			p.renderParagraph(data[:i])
			return i
		}

		// if there's a fenced code block, paragraph is over
		if p.extensions&FencedCode != 0 {
			if p.fencedCodeBlock(current, false) > 0 {
				p.renderParagraph(data[:i])
				return i
			}
		}

		// if there's a definition list item, prev line is a definition term
		if p.extensions&DefinitionLists != 0 {
			if p.dliPrefix(current) != 0 {
				ret := p.list(data[prev:], ListTypeDefinition)
				return ret
			}
		}

		// if there's a list after this, paragraph is over
		if p.extensions&NoEmptyLineBeforeBlock != 0 {
			if p.uliPrefix(current) != 0 ||
				p.oliPrefix(current) != 0 ||
				p.quotePrefix(current) != 0 ||
				p.codePrefix(current) != 0 {
				p.renderParagraph(data[:i])
				return i
			}
		}

		// otherwise, scan to the beginning of the next line
		nl := bytes.IndexByte(data[i:], '\n')
		if nl >= 0 {
			i += nl + 1
		} else {
			i += len(data[i:])
		}
	}

	p.renderParagraph(data[:i])
	return i
}

func skipChar(data []byte, start int, char byte) int {
	i := start
	for i < len(data) && data[i] == char {
		i++
	}
	return i
}

func skipUntilChar(text []byte, start int, char byte) int {
	i := start
	for i < len(text) && text[i] != char {
		i++
	}
	return i
}
