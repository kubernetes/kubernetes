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
	"strings"
	"unicode"
)

// Parse block-level data.
// Note: this function and many that it calls assume that
// the input buffer ends with a newline.
func (p *parser) block(out *bytes.Buffer, data []byte) {
	if len(data) == 0 || data[len(data)-1] != '\n' {
		panic("block input is missing terminating newline")
	}

	// this is called recursively: enforce a maximum depth
	if p.nesting >= p.maxNesting {
		return
	}
	p.nesting++

	// parse out one block-level construct at a time
	for len(data) > 0 {
		// prefixed header:
		//
		// # Header 1
		// ## Header 2
		// ...
		// ###### Header 6
		if p.isPrefixHeader(data) {
			data = data[p.prefixHeader(out, data):]
			continue
		}

		// block of preformatted HTML:
		//
		// <div>
		//     ...
		// </div>
		if data[0] == '<' {
			if i := p.html(out, data, true); i > 0 {
				data = data[i:]
				continue
			}
		}

		// title block
		//
		// % stuff
		// % more stuff
		// % even more stuff
		if p.flags&EXTENSION_TITLEBLOCK != 0 {
			if data[0] == '%' {
				if i := p.titleBlock(out, data, true); i > 0 {
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
			data = data[p.code(out, data):]
			continue
		}

		// fenced code block:
		//
		// ``` go info string here
		// func fact(n int) int {
		//     if n <= 1 {
		//         return n
		//     }
		//     return n * fact(n-1)
		// }
		// ```
		if p.flags&EXTENSION_FENCED_CODE != 0 {
			if i := p.fencedCodeBlock(out, data, true); i > 0 {
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
			p.r.HRule(out)
			var i int
			for i = 0; data[i] != '\n'; i++ {
			}
			data = data[i:]
			continue
		}

		// block quote:
		//
		// > A big quote I found somewhere
		// > on the web
		if p.quotePrefix(data) > 0 {
			data = data[p.quote(out, data):]
			continue
		}

		// table:
		//
		// Name  | Age | Phone
		// ------|-----|---------
		// Bob   | 31  | 555-1234
		// Alice | 27  | 555-4321
		if p.flags&EXTENSION_TABLES != 0 {
			if i := p.table(out, data); i > 0 {
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
			data = data[p.list(out, data, 0):]
			continue
		}

		// a numbered/ordered list:
		//
		// 1. Item 1
		// 2. Item 2
		if p.oliPrefix(data) > 0 {
			data = data[p.list(out, data, LIST_TYPE_ORDERED):]
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
		if p.flags&EXTENSION_DEFINITION_LISTS != 0 {
			if p.dliPrefix(data) > 0 {
				data = data[p.list(out, data, LIST_TYPE_DEFINITION):]
				continue
			}
		}

		// anything else must look like a normal paragraph
		// note: this finds underlined headers, too
		data = data[p.paragraph(out, data):]
	}

	p.nesting--
}

func (p *parser) isPrefixHeader(data []byte) bool {
	if data[0] != '#' {
		return false
	}

	if p.flags&EXTENSION_SPACE_HEADERS != 0 {
		level := 0
		for level < 6 && data[level] == '#' {
			level++
		}
		if data[level] != ' ' {
			return false
		}
	}
	return true
}

func (p *parser) prefixHeader(out *bytes.Buffer, data []byte) int {
	level := 0
	for level < 6 && data[level] == '#' {
		level++
	}
	i := skipChar(data, level, ' ')
	end := skipUntilChar(data, i, '\n')
	skip := end
	id := ""
	if p.flags&EXTENSION_HEADER_IDS != 0 {
		j, k := 0, 0
		// find start/end of header id
		for j = i; j < end-1 && (data[j] != '{' || data[j+1] != '#'); j++ {
		}
		for k = j + 1; k < end && data[k] != '}'; k++ {
		}
		// extract header id iff found
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
		if id == "" && p.flags&EXTENSION_AUTO_HEADER_IDS != 0 {
			id = SanitizedAnchorName(string(data[i:end]))
		}
		work := func() bool {
			p.inline(out, data[i:end])
			return true
		}
		p.r.Header(out, work, level, id)
	}
	return skip
}

func (p *parser) isUnderlinedHeader(data []byte) int {
	// test of level 1 header
	if data[0] == '=' {
		i := skipChar(data, 1, '=')
		i = skipChar(data, i, ' ')
		if data[i] == '\n' {
			return 1
		} else {
			return 0
		}
	}

	// test of level 2 header
	if data[0] == '-' {
		i := skipChar(data, 1, '-')
		i = skipChar(data, i, ' ')
		if data[i] == '\n' {
			return 2
		} else {
			return 0
		}
	}

	return 0
}

func (p *parser) titleBlock(out *bytes.Buffer, data []byte, doRender bool) int {
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
	p.r.TitleBlock(out, data)

	return len(data)
}

func (p *parser) html(out *bytes.Buffer, data []byte, doRender bool) int {
	var i, j int

	// identify the opening tag
	if data[0] != '<' {
		return 0
	}
	curtag, tagfound := p.htmlFindTag(data[1:])

	// handle special cases
	if !tagfound {
		// check for an HTML comment
		if size := p.htmlComment(out, data, doRender); size > 0 {
			return size
		}

		// check for an <hr> tag
		if size := p.htmlHr(out, data, doRender); size > 0 {
			return size
		}

		// check for HTML CDATA
		if size := p.htmlCDATA(out, data, doRender); size > 0 {
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
		p.r.BlockHtml(out, data[:end])
	}

	return i
}

func (p *parser) renderHTMLBlock(out *bytes.Buffer, data []byte, start int, doRender bool) int {
	// html block needs to end with a blank line
	if i := p.isEmpty(data[start:]); i > 0 {
		size := start + i
		if doRender {
			// trim trailing newlines
			end := size
			for end > 0 && data[end-1] == '\n' {
				end--
			}
			p.r.BlockHtml(out, data[:end])
		}
		return size
	}
	return 0
}

// HTML comment, lax form
func (p *parser) htmlComment(out *bytes.Buffer, data []byte, doRender bool) int {
	i := p.inlineHTMLComment(out, data)
	return p.renderHTMLBlock(out, data, i, doRender)
}

// HTML CDATA section
func (p *parser) htmlCDATA(out *bytes.Buffer, data []byte, doRender bool) int {
	const cdataTag = "<![cdata["
	const cdataTagLen = len(cdataTag)
	if len(data) < cdataTagLen+1 {
		return 0
	}
	if !bytes.Equal(bytes.ToLower(data[:cdataTagLen]), []byte(cdataTag)) {
		return 0
	}
	i := cdataTagLen
	// scan for an end-of-comment marker, across lines if necessary
	for i < len(data) && !(data[i-2] == ']' && data[i-1] == ']' && data[i] == '>') {
		i++
	}
	i++
	// no end-of-comment marker
	if i >= len(data) {
		return 0
	}
	return p.renderHTMLBlock(out, data, i, doRender)
}

// HR, which is the only self-closing block tag considered
func (p *parser) htmlHr(out *bytes.Buffer, data []byte, doRender bool) int {
	if data[0] != '<' || (data[1] != 'h' && data[1] != 'H') || (data[2] != 'r' && data[2] != 'R') {
		return 0
	}
	if data[3] != ' ' && data[3] != '/' && data[3] != '>' {
		// not an <hr> tag after all; at least not a valid one
		return 0
	}

	i := 3
	for data[i] != '>' && data[i] != '\n' {
		i++
	}

	if data[i] == '>' {
		return p.renderHTMLBlock(out, data, i+1, doRender)
	}

	return 0
}

func (p *parser) htmlFindTag(data []byte) (string, bool) {
	i := 0
	for isalnum(data[i]) {
		i++
	}
	key := string(data[:i])
	if _, ok := blockTags[key]; ok {
		return key, true
	}
	return "", false
}

func (p *parser) htmlFindEnd(tag string, data []byte) int {
	// assume data[0] == '<' && data[1] == '/' already tested

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

	if p.flags&EXTENSION_LAX_HTML_BLOCKS != 0 {
		return i
	}
	if skip = p.isEmpty(data[i:]); skip == 0 {
		// following line must be blank
		return 0
	}

	return i + skip
}

func (*parser) isEmpty(data []byte) int {
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
	return i + 1
}

func (*parser) isHRule(data []byte) bool {
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
	for data[i] != '\n' {
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
// A final newline is mandatory to recognize the fence line, unless newlineOptional is true.
func isFenceLine(data []byte, info *string, oldmarker string, newlineOptional bool) (end int, marker string) {
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
	// into one, always get the info string, and discard it if the caller doesn't care.
	if info != nil {
		infoLength := 0
		i = skipChar(data, i, ' ')

		if i >= len(data) {
			if newlineOptional && i == len(data) {
				return i, marker
			}
			return 0, ""
		}

		infoStart := i

		if data[i] == '{' {
			i++
			infoStart++

			for i < len(data) && data[i] != '}' && data[i] != '\n' {
				infoLength++
				i++
			}

			if i >= len(data) || data[i] != '}' {
				return 0, ""
			}

			// strip all whitespace at the beginning and the end
			// of the {} block
			for infoLength > 0 && isspace(data[infoStart]) {
				infoStart++
				infoLength--
			}

			for infoLength > 0 && isspace(data[infoStart+infoLength-1]) {
				infoLength--
			}

			i++
		} else {
			for i < len(data) && !isverticalspace(data[i]) {
				infoLength++
				i++
			}
		}

		*info = strings.TrimSpace(string(data[infoStart : infoStart+infoLength]))
	}

	i = skipChar(data, i, ' ')
	if i >= len(data) || data[i] != '\n' {
		if newlineOptional && i == len(data) {
			return i, marker
		}
		return 0, ""
	}

	return i + 1, marker // Take newline into account.
}

// fencedCodeBlock returns the end index if data contains a fenced code block at the beginning,
// or 0 otherwise. It writes to out if doRender is true, otherwise it has no side effects.
// If doRender is true, a final newline is mandatory to recognize the fenced code block.
func (p *parser) fencedCodeBlock(out *bytes.Buffer, data []byte, doRender bool) int {
	var infoString string
	beg, marker := isFenceLine(data, &infoString, "", false)
	if beg == 0 || beg >= len(data) {
		return 0
	}

	var work bytes.Buffer

	for {
		// safe to assume beg < len(data)

		// check for the end of the code block
		newlineOptional := !doRender
		fenceEnd, _ := isFenceLine(data[beg:], nil, marker, newlineOptional)
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
		p.r.BlockCode(out, work.Bytes(), infoString)
	}

	return beg
}

func (p *parser) table(out *bytes.Buffer, data []byte) int {
	var header bytes.Buffer
	i, columns := p.tableHeader(&header, data)
	if i == 0 {
		return 0
	}

	var body bytes.Buffer

	for i < len(data) {
		pipes, rowStart := 0, i
		for ; data[i] != '\n'; i++ {
			if data[i] == '|' {
				pipes++
			}
		}

		if pipes == 0 {
			i = rowStart
			break
		}

		// include the newline in data sent to tableRow
		i++
		p.tableRow(&body, data[rowStart:i], columns, false)
	}

	p.r.Table(out, header.Bytes(), body.Bytes(), columns)

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

func (p *parser) tableHeader(out *bytes.Buffer, data []byte) (size int, columns []int) {
	i := 0
	colCount := 1
	for i = 0; data[i] != '\n'; i++ {
		if data[i] == '|' && !isBackslashEscaped(data, i) {
			colCount++
		}
	}

	// doesn't look like a table header
	if colCount == 1 {
		return
	}

	// include the newline in the data sent to tableRow
	header := data[:i+1]

	// column count ignores pipes at beginning or end of line
	if data[0] == '|' {
		colCount--
	}
	if i > 2 && data[i-1] == '|' && !isBackslashEscaped(data, i-1) {
		colCount--
	}

	columns = make([]int, colCount)

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
	for data[i] != '\n' {
		dashes := 0

		if data[i] == ':' {
			i++
			columns[col] |= TABLE_ALIGNMENT_LEFT
			dashes++
		}
		for data[i] == '-' {
			i++
			dashes++
		}
		if data[i] == ':' {
			i++
			columns[col] |= TABLE_ALIGNMENT_RIGHT
			dashes++
		}
		for data[i] == ' ' {
			i++
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
			for data[i] == ' ' {
				i++
			}

			// trailing junk found after last column
			if col >= colCount && data[i] != '\n' {
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

	p.tableRow(out, header, columns, true)
	size = i + 1
	return
}

func (p *parser) tableRow(out *bytes.Buffer, data []byte, columns []int, header bool) {
	i, col := 0, 0
	var rowWork bytes.Buffer

	if data[i] == '|' && !isBackslashEscaped(data, i) {
		i++
	}

	for col = 0; col < len(columns) && i < len(data); col++ {
		for data[i] == ' ' {
			i++
		}

		cellStart := i

		for (data[i] != '|' || isBackslashEscaped(data, i)) && data[i] != '\n' {
			i++
		}

		cellEnd := i

		// skip the end-of-cell marker, possibly taking us past end of buffer
		i++

		for cellEnd > cellStart && data[cellEnd-1] == ' ' {
			cellEnd--
		}

		var cellWork bytes.Buffer
		p.inline(&cellWork, data[cellStart:cellEnd])

		if header {
			p.r.TableHeaderCell(&rowWork, cellWork.Bytes(), columns[col])
		} else {
			p.r.TableCell(&rowWork, cellWork.Bytes(), columns[col])
		}
	}

	// pad it out with empty columns to get the right number
	for ; col < len(columns); col++ {
		if header {
			p.r.TableHeaderCell(&rowWork, nil, columns[col])
		} else {
			p.r.TableCell(&rowWork, nil, columns[col])
		}
	}

	// silently ignore rows with too many cells

	p.r.TableRow(out, rowWork.Bytes())
}

// returns blockquote prefix length
func (p *parser) quotePrefix(data []byte) int {
	i := 0
	for i < 3 && data[i] == ' ' {
		i++
	}
	if data[i] == '>' {
		if data[i+1] == ' ' {
			return i + 2
		}
		return i + 1
	}
	return 0
}

// blockquote ends with at least one blank line
// followed by something without a blockquote prefix
func (p *parser) terminateBlockquote(data []byte, beg, end int) bool {
	if p.isEmpty(data[beg:]) <= 0 {
		return false
	}
	if end >= len(data) {
		return true
	}
	return p.quotePrefix(data[end:]) == 0 && p.isEmpty(data[end:]) == 0
}

// parse a blockquote fragment
func (p *parser) quote(out *bytes.Buffer, data []byte) int {
	var raw bytes.Buffer
	beg, end := 0, 0
	for beg < len(data) {
		end = beg
		// Step over whole lines, collecting them. While doing that, check for
		// fenced code and if one's found, incorporate it altogether,
		// irregardless of any contents inside it
		for data[end] != '\n' {
			if p.flags&EXTENSION_FENCED_CODE != 0 {
				if i := p.fencedCodeBlock(out, data[end:], false); i > 0 {
					// -1 to compensate for the extra end++ after the loop:
					end += i - 1
					break
				}
			}
			end++
		}
		end++

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

	var cooked bytes.Buffer
	p.block(&cooked, raw.Bytes())
	p.r.BlockQuote(out, cooked.Bytes())
	return end
}

// returns prefix length for block code
func (p *parser) codePrefix(data []byte) int {
	if data[0] == ' ' && data[1] == ' ' && data[2] == ' ' && data[3] == ' ' {
		return 4
	}
	return 0
}

func (p *parser) code(out *bytes.Buffer, data []byte) int {
	var work bytes.Buffer

	i := 0
	for i < len(data) {
		beg := i
		for data[i] != '\n' {
			i++
		}
		i++

		blankline := p.isEmpty(data[beg:i]) > 0
		if pre := p.codePrefix(data[beg:i]); pre > 0 {
			beg += pre
		} else if !blankline {
			// non-empty, non-prefixed line breaks the pre
			i = beg
			break
		}

		// verbatim copy to the working buffeu
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

	p.r.BlockCode(out, work.Bytes(), "")

	return i
}

// returns unordered list item prefix
func (p *parser) uliPrefix(data []byte) int {
	i := 0

	// start with up to 3 spaces
	for i < 3 && data[i] == ' ' {
		i++
	}

	// need a *, +, or - followed by a space
	if (data[i] != '*' && data[i] != '+' && data[i] != '-') ||
		data[i+1] != ' ' {
		return 0
	}
	return i + 2
}

// returns ordered list item prefix
func (p *parser) oliPrefix(data []byte) int {
	i := 0

	// start with up to 3 spaces
	for i < 3 && data[i] == ' ' {
		i++
	}

	// count the digits
	start := i
	for data[i] >= '0' && data[i] <= '9' {
		i++
	}

	// we need >= 1 digits followed by a dot and a space
	if start == i || data[i] != '.' || data[i+1] != ' ' {
		return 0
	}
	return i + 2
}

// returns definition list item prefix
func (p *parser) dliPrefix(data []byte) int {
	i := 0

	// need a : followed by a spaces
	if data[i] != ':' || data[i+1] != ' ' {
		return 0
	}
	for data[i] == ' ' {
		i++
	}
	return i + 2
}

// parse ordered or unordered list block
func (p *parser) list(out *bytes.Buffer, data []byte, flags int) int {
	i := 0
	flags |= LIST_ITEM_BEGINNING_OF_LIST
	work := func() bool {
		for i < len(data) {
			skip := p.listItem(out, data[i:], &flags)
			i += skip

			if skip == 0 || flags&LIST_ITEM_END_OF_LIST != 0 {
				break
			}
			flags &= ^LIST_ITEM_BEGINNING_OF_LIST
		}
		return true
	}

	p.r.List(out, work, flags)
	return i
}

// Parse a single list item.
// Assumes initial prefix is already removed if this is a sublist.
func (p *parser) listItem(out *bytes.Buffer, data []byte, flags *int) int {
	// keep track of the indentation of the first line
	itemIndent := 0
	for itemIndent < 3 && data[itemIndent] == ' ' {
		itemIndent++
	}

	i := p.uliPrefix(data)
	if i == 0 {
		i = p.oliPrefix(data)
	}
	if i == 0 {
		i = p.dliPrefix(data)
		// reset definition term flag
		if i > 0 {
			*flags &= ^LIST_TYPE_TERM
		}
	}
	if i == 0 {
		// if in defnition list, set term flag and continue
		if *flags&LIST_TYPE_DEFINITION != 0 {
			*flags |= LIST_TYPE_TERM
		} else {
			return 0
		}
	}

	// skip leading whitespace on first line
	for data[i] == ' ' {
		i++
	}

	// find the end of the line
	line := i
	for i > 0 && data[i-1] != '\n' {
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
	codeBlockMarker := ""

gatherlines:
	for line < len(data) {
		i++

		// find the end of this line
		for data[i-1] != '\n' {
			i++
		}

		// if it is an empty line, guess that it is part of this item
		// and move on to the next line
		if p.isEmpty(data[line:i]) > 0 {
			containsBlankLine = true
			raw.Write(data[line:i])
			line = i
			continue
		}

		// calculate the indentation
		indent := 0
		for indent < 4 && line+indent < i && data[line+indent] == ' ' {
			indent++
		}

		chunk := data[line+indent : i]

		if p.flags&EXTENSION_FENCED_CODE != 0 {
			// determine if in or out of codeblock
			// if in codeblock, ignore normal list processing
			_, marker := isFenceLine(chunk, nil, codeBlockMarker, false)
			if marker != "" {
				if codeBlockMarker == "" {
					// start of codeblock
					codeBlockMarker = marker
				} else {
					// end of codeblock.
					*flags |= LIST_ITEM_CONTAINS_BLOCK
					codeBlockMarker = ""
				}
			}
			// we are in a codeblock, write line, and continue
			if codeBlockMarker != "" || marker != "" {
				raw.Write(data[line+indent : i])
				line = i
				continue gatherlines
			}
		}

		// evaluate how this line fits in
		switch {
		// is this a nested list item?
		case (p.uliPrefix(chunk) > 0 && !p.isHRule(chunk)) ||
			p.oliPrefix(chunk) > 0 ||
			p.dliPrefix(chunk) > 0:

			if containsBlankLine {
				// end the list if the type changed after a blank line
				if indent <= itemIndent &&
					((*flags&LIST_TYPE_ORDERED != 0 && p.uliPrefix(chunk) > 0) ||
						(*flags&LIST_TYPE_ORDERED == 0 && p.oliPrefix(chunk) > 0)) {

					*flags |= LIST_ITEM_END_OF_LIST
					break gatherlines
				}
				*flags |= LIST_ITEM_CONTAINS_BLOCK
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

		// is this a nested prefix header?
		case p.isPrefixHeader(chunk):
			// if the header is not indented, it is not nested in the list
			// and thus ends the list
			if containsBlankLine && indent < 4 {
				*flags |= LIST_ITEM_END_OF_LIST
				break gatherlines
			}
			*flags |= LIST_ITEM_CONTAINS_BLOCK

		// anything following an empty line is only part
		// of this item if it is indented 4 spaces
		// (regardless of the indentation of the beginning of the item)
		case containsBlankLine && indent < 4:
			if *flags&LIST_TYPE_DEFINITION != 0 && i < len(data)-1 {
				// is the next item still a part of this list?
				next := i
				for data[next] != '\n' {
					next++
				}
				for next < len(data)-1 && data[next] == '\n' {
					next++
				}
				if i < len(data)-1 && data[i] != ':' && data[next] != ':' {
					*flags |= LIST_ITEM_END_OF_LIST
				}
			} else {
				*flags |= LIST_ITEM_END_OF_LIST
			}
			break gatherlines

		// a blank line means this should be parsed as a block
		case containsBlankLine:
			*flags |= LIST_ITEM_CONTAINS_BLOCK
		}

		containsBlankLine = false

		// add the line into the working buffer without prefix
		raw.Write(data[line+indent : i])

		line = i
	}

	// If reached end of data, the Renderer.ListItem call we're going to make below
	// is definitely the last in the list.
	if line >= len(data) {
		*flags |= LIST_ITEM_END_OF_LIST
	}

	rawBytes := raw.Bytes()

	// render the contents of the list item
	var cooked bytes.Buffer
	if *flags&LIST_ITEM_CONTAINS_BLOCK != 0 && *flags&LIST_TYPE_TERM == 0 {
		// intermediate render of block item, except for definition term
		if sublist > 0 {
			p.block(&cooked, rawBytes[:sublist])
			p.block(&cooked, rawBytes[sublist:])
		} else {
			p.block(&cooked, rawBytes)
		}
	} else {
		// intermediate render of inline item
		if sublist > 0 {
			p.inline(&cooked, rawBytes[:sublist])
			p.block(&cooked, rawBytes[sublist:])
		} else {
			p.inline(&cooked, rawBytes)
		}
	}

	// render the actual list item
	cookedBytes := cooked.Bytes()
	parsedEnd := len(cookedBytes)

	// strip trailing newlines
	for parsedEnd > 0 && cookedBytes[parsedEnd-1] == '\n' {
		parsedEnd--
	}
	p.r.ListItem(out, cookedBytes[:parsedEnd], *flags)

	return line
}

// render a single paragraph that has already been parsed out
func (p *parser) renderParagraph(out *bytes.Buffer, data []byte) {
	if len(data) == 0 {
		return
	}

	// trim leading spaces
	beg := 0
	for data[beg] == ' ' {
		beg++
	}

	// trim trailing newline
	end := len(data) - 1

	// trim trailing spaces
	for end > beg && data[end-1] == ' ' {
		end--
	}

	work := func() bool {
		p.inline(out, data[beg:end])
		return true
	}
	p.r.Paragraph(out, work)
}

func (p *parser) paragraph(out *bytes.Buffer, data []byte) int {
	// prev: index of 1st char of previous line
	// line: index of 1st char of current line
	// i: index of cursor/end of current line
	var prev, line, i int

	// keep going until we find something to mark the end of the paragraph
	for i < len(data) {
		// mark the beginning of the current line
		prev = line
		current := data[i:]
		line = i

		// did we find a blank line marking the end of the paragraph?
		if n := p.isEmpty(current); n > 0 {
			// did this blank line followed by a definition list item?
			if p.flags&EXTENSION_DEFINITION_LISTS != 0 {
				if i < len(data)-1 && data[i+1] == ':' {
					return p.list(out, data[prev:], LIST_TYPE_DEFINITION)
				}
			}

			p.renderParagraph(out, data[:i])
			return i + n
		}

		// an underline under some text marks a header, so our paragraph ended on prev line
		if i > 0 {
			if level := p.isUnderlinedHeader(current); level > 0 {
				// render the paragraph
				p.renderParagraph(out, data[:prev])

				// ignore leading and trailing whitespace
				eol := i - 1
				for prev < eol && data[prev] == ' ' {
					prev++
				}
				for eol > prev && data[eol-1] == ' ' {
					eol--
				}

				// render the header
				// this ugly double closure avoids forcing variables onto the heap
				work := func(o *bytes.Buffer, pp *parser, d []byte) func() bool {
					return func() bool {
						pp.inline(o, d)
						return true
					}
				}(out, p, data[prev:eol])

				id := ""
				if p.flags&EXTENSION_AUTO_HEADER_IDS != 0 {
					id = SanitizedAnchorName(string(data[prev:eol]))
				}

				p.r.Header(out, work, level, id)

				// find the end of the underline
				for data[i] != '\n' {
					i++
				}
				return i
			}
		}

		// if the next line starts a block of HTML, then the paragraph ends here
		if p.flags&EXTENSION_LAX_HTML_BLOCKS != 0 {
			if data[i] == '<' && p.html(out, current, false) > 0 {
				// rewind to before the HTML block
				p.renderParagraph(out, data[:i])
				return i
			}
		}

		// if there's a prefixed header or a horizontal rule after this, paragraph is over
		if p.isPrefixHeader(current) || p.isHRule(current) {
			p.renderParagraph(out, data[:i])
			return i
		}

		// if there's a fenced code block, paragraph is over
		if p.flags&EXTENSION_FENCED_CODE != 0 {
			if p.fencedCodeBlock(out, current, false) > 0 {
				p.renderParagraph(out, data[:i])
				return i
			}
		}

		// if there's a definition list item, prev line is a definition term
		if p.flags&EXTENSION_DEFINITION_LISTS != 0 {
			if p.dliPrefix(current) != 0 {
				return p.list(out, data[prev:], LIST_TYPE_DEFINITION)
			}
		}

		// if there's a list after this, paragraph is over
		if p.flags&EXTENSION_NO_EMPTY_LINE_BEFORE_BLOCK != 0 {
			if p.uliPrefix(current) != 0 ||
				p.oliPrefix(current) != 0 ||
				p.quotePrefix(current) != 0 ||
				p.codePrefix(current) != 0 {
				p.renderParagraph(out, data[:i])
				return i
			}
		}

		// otherwise, scan to the beginning of the next line
		for data[i] != '\n' {
			i++
		}
		i++
	}

	p.renderParagraph(out, data[:i])
	return i
}

// SanitizedAnchorName returns a sanitized anchor name for the given text.
//
// It implements the algorithm specified in the package comment.
func SanitizedAnchorName(text string) string {
	var anchorName []rune
	futureDash := false
	for _, r := range text {
		switch {
		case unicode.IsLetter(r) || unicode.IsNumber(r):
			if futureDash && len(anchorName) > 0 {
				anchorName = append(anchorName, '-')
			}
			futureDash = false
			anchorName = append(anchorName, unicode.ToLower(r))
		default:
			futureDash = true
		}
	}
	return string(anchorName)
}
