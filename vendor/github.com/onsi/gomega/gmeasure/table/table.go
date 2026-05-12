package table

// This is a temporary package - Table will move to github.com/onsi/consolable once some more dust settles

import (
	"reflect"
	"strings"
	"unicode/utf8"
)

type AlignType uint

const (
	AlignTypeLeft AlignType = iota
	AlignTypeCenter
	AlignTypeRight
)

type Divider string

type Row struct {
	Cells   []Cell
	Divider string
	Style   string
}

func R(args ...any) *Row {
	r := &Row{
		Divider: "-",
	}
	for _, arg := range args {
		switch reflect.TypeOf(arg) {
		case reflect.TypeOf(Divider("")):
			r.Divider = string(arg.(Divider))
		case reflect.TypeOf(r.Style):
			r.Style = arg.(string)
		case reflect.TypeOf(Cell{}):
			r.Cells = append(r.Cells, arg.(Cell))
		}
	}
	return r
}

func (r *Row) AppendCell(cells ...Cell) *Row {
	r.Cells = append(r.Cells, cells...)
	return r
}

func (r *Row) Render(widths []int, totalWidth int, tableStyle TableStyle, isLastRow bool) string {
	out := ""
	if len(r.Cells) == 1 {
		out += strings.Join(r.Cells[0].render(totalWidth, r.Style, tableStyle), "\n") + "\n"
	} else {
		if len(r.Cells) != len(widths) {
			panic("row vs width mismatch")
		}
		renderedCells := make([][]string, len(r.Cells))
		maxHeight := 0
		for colIdx, cell := range r.Cells {
			renderedCells[colIdx] = cell.render(widths[colIdx], r.Style, tableStyle)
			if len(renderedCells[colIdx]) > maxHeight {
				maxHeight = len(renderedCells[colIdx])
			}
		}
		for colIdx := range r.Cells {
			for len(renderedCells[colIdx]) < maxHeight {
				renderedCells[colIdx] = append(renderedCells[colIdx], strings.Repeat(" ", widths[colIdx]))
			}
		}
		border := strings.Repeat(" ", tableStyle.Padding)
		if tableStyle.VerticalBorders {
			border += "|" + border
		}
		for lineIdx := 0; lineIdx < maxHeight; lineIdx++ {
			for colIdx := range r.Cells {
				out += renderedCells[colIdx][lineIdx]
				if colIdx < len(r.Cells)-1 {
					out += border
				}
			}
			out += "\n"
		}
	}
	if tableStyle.HorizontalBorders && !isLastRow && r.Divider != "" {
		out += strings.Repeat(string(r.Divider), totalWidth) + "\n"
	}

	return out
}

type Cell struct {
	Contents []string
	Style    string
	Align    AlignType
}

func C(contents string, args ...any) Cell {
	c := Cell{
		Contents: strings.Split(contents, "\n"),
	}
	for _, arg := range args {
		switch reflect.TypeOf(arg) {
		case reflect.TypeOf(c.Style):
			c.Style = arg.(string)
		case reflect.TypeOf(c.Align):
			c.Align = arg.(AlignType)
		}
	}
	return c
}

func (c Cell) Width() (int, int) {
	w, minW := 0, 0
	for _, line := range c.Contents {
		lineWidth := utf8.RuneCountInString(line)
		if lineWidth > w {
			w = lineWidth
		}
		for _, word := range strings.Split(line, " ") {
			wordWidth := utf8.RuneCountInString(word)
			if wordWidth > minW {
				minW = wordWidth
			}
		}
	}
	return w, minW
}

func (c Cell) alignLine(line string, width int) string {
	lineWidth := utf8.RuneCountInString(line)
	if lineWidth == width {
		return line
	}
	if lineWidth < width {
		gap := width - lineWidth
		switch c.Align {
		case AlignTypeLeft:
			return line + strings.Repeat(" ", gap)
		case AlignTypeRight:
			return strings.Repeat(" ", gap) + line
		case AlignTypeCenter:
			leftGap := gap / 2
			rightGap := gap - leftGap
			return strings.Repeat(" ", leftGap) + line + strings.Repeat(" ", rightGap)
		}
	}
	return line
}

func (c Cell) splitWordToWidth(word string, width int) []string {
	out := []string{}
	n, subWord := 0, ""
	for _, c := range word {
		subWord += string(c)
		n += 1
		if n == width-1 {
			out = append(out, subWord+"-")
			n, subWord = 0, ""
		}
	}
	return out
}

func (c Cell) splitToWidth(line string, width int) []string {
	lineWidth := utf8.RuneCountInString(line)
	if lineWidth <= width {
		return []string{line}
	}

	outLines := []string{}
	words := strings.Split(line, " ")
	outWords := []string{words[0]}
	length := utf8.RuneCountInString(words[0])
	if length > width {
		splitWord := c.splitWordToWidth(words[0], width)
		lastIdx := len(splitWord) - 1
		outLines = append(outLines, splitWord[:lastIdx]...)
		outWords = []string{splitWord[lastIdx]}
		length = utf8.RuneCountInString(splitWord[lastIdx])
	}

	for _, word := range words[1:] {
		wordLength := utf8.RuneCountInString(word)
		if length+wordLength+1 <= width {
			length += wordLength + 1
			outWords = append(outWords, word)
			continue
		}
		outLines = append(outLines, strings.Join(outWords, " "))

		outWords = []string{word}
		length = wordLength
		if length > width {
			splitWord := c.splitWordToWidth(word, width)
			lastIdx := len(splitWord) - 1
			outLines = append(outLines, splitWord[:lastIdx]...)
			outWords = []string{splitWord[lastIdx]}
			length = utf8.RuneCountInString(splitWord[lastIdx])
		}
	}
	if len(outWords) > 0 {
		outLines = append(outLines, strings.Join(outWords, " "))
	}

	return outLines
}

func (c Cell) render(width int, style string, tableStyle TableStyle) []string {
	out := []string{}
	for _, line := range c.Contents {
		out = append(out, c.splitToWidth(line, width)...)
	}
	for idx := range out {
		out[idx] = c.alignLine(out[idx], width)
	}

	if tableStyle.EnableTextStyling {
		style = style + c.Style
		if style != "" {
			for idx := range out {
				out[idx] = style + out[idx] + "{{/}}"
			}
		}
	}

	return out
}

type TableStyle struct {
	Padding           int
	VerticalBorders   bool
	HorizontalBorders bool
	MaxTableWidth     int
	MaxColWidth       int
	EnableTextStyling bool
}

var DefaultTableStyle = TableStyle{
	Padding:           1,
	VerticalBorders:   true,
	HorizontalBorders: true,
	MaxTableWidth:     120,
	MaxColWidth:       40,
	EnableTextStyling: true,
}

type Table struct {
	Rows []*Row

	TableStyle TableStyle
}

func NewTable() *Table {
	return &Table{
		TableStyle: DefaultTableStyle,
	}
}

func (t *Table) AppendRow(row *Row) *Table {
	t.Rows = append(t.Rows, row)
	return t
}

func (t *Table) Render() string {
	out := ""
	totalWidth, widths := t.computeWidths()
	for rowIdx, row := range t.Rows {
		out += row.Render(widths, totalWidth, t.TableStyle, rowIdx == len(t.Rows)-1)
	}
	return out
}

func (t *Table) computeWidths() (int, []int) {
	nCol := 0
	for _, row := range t.Rows {
		if len(row.Cells) > nCol {
			nCol = len(row.Cells)
		}
	}

	// lets compute the contribution to width from the borders + padding
	borderWidth := t.TableStyle.Padding
	if t.TableStyle.VerticalBorders {
		borderWidth += 1 + t.TableStyle.Padding
	}
	totalBorderWidth := borderWidth * (nCol - 1)

	// lets compute the width of each column
	widths := make([]int, nCol)
	minWidths := make([]int, nCol)
	for colIdx := range widths {
		for _, row := range t.Rows {
			if colIdx >= len(row.Cells) {
				// ignore rows with fewer columns
				continue
			}
			w, minWid := row.Cells[colIdx].Width()
			if w > widths[colIdx] {
				widths[colIdx] = w
			}
			if minWid > minWidths[colIdx] {
				minWidths[colIdx] = minWid
			}
		}
	}

	// do we already fit?
	if sum(widths)+totalBorderWidth <= t.TableStyle.MaxTableWidth {
		// yes! we're done
		return sum(widths) + totalBorderWidth, widths
	}

	// clamp the widths and minWidths to MaxColWidth
	for colIdx := range widths {
		widths[colIdx] = min(widths[colIdx], t.TableStyle.MaxColWidth)
		minWidths[colIdx] = min(minWidths[colIdx], t.TableStyle.MaxColWidth)
	}

	// do we fit now?
	if sum(widths)+totalBorderWidth <= t.TableStyle.MaxTableWidth {
		// yes! we're done
		return sum(widths) + totalBorderWidth, widths
	}

	// hmm... still no... can we possibly squeeze the table in without violating minWidths?
	if sum(minWidths)+totalBorderWidth >= t.TableStyle.MaxTableWidth {
		// nope - we're just going to have to exceed MaxTableWidth
		return sum(minWidths) + totalBorderWidth, minWidths
	}

	// looks like we don't fit yet, but we should be able to fit without violating minWidths
	// lets start scaling down
	n := 0
	for sum(widths)+totalBorderWidth > t.TableStyle.MaxTableWidth {
		budget := t.TableStyle.MaxTableWidth - totalBorderWidth
		baseline := sum(widths)

		for colIdx := range widths {
			widths[colIdx] = max((widths[colIdx]*budget)/baseline, minWidths[colIdx])
		}
		n += 1
		if n > 100 {
			break // in case we somehow fail to converge
		}
	}

	return sum(widths) + totalBorderWidth, widths
}

func sum(s []int) int {
	out := 0
	for _, v := range s {
		out += v
	}
	return out
}
