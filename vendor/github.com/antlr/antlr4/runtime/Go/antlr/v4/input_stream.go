// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

type InputStream struct {
	name  string
	index int
	data  []rune
	size  int
}

func NewInputStream(data string) *InputStream {

	is := new(InputStream)

	is.name = "<empty>"
	is.index = 0
	is.data = []rune(data)
	is.size = len(is.data) // number of runes

	return is
}

func (is *InputStream) reset() {
	is.index = 0
}

func (is *InputStream) Consume() {
	if is.index >= is.size {
		// assert is.LA(1) == TokenEOF
		panic("cannot consume EOF")
	}
	is.index++
}

func (is *InputStream) LA(offset int) int {

	if offset == 0 {
		return 0 // nil
	}
	if offset < 0 {
		offset++ // e.g., translate LA(-1) to use offset=0
	}
	pos := is.index + offset - 1

	if pos < 0 || pos >= is.size { // invalid
		return TokenEOF
	}

	return int(is.data[pos])
}

func (is *InputStream) LT(offset int) int {
	return is.LA(offset)
}

func (is *InputStream) Index() int {
	return is.index
}

func (is *InputStream) Size() int {
	return is.size
}

// mark/release do nothing we have entire buffer
func (is *InputStream) Mark() int {
	return -1
}

func (is *InputStream) Release(marker int) {
}

func (is *InputStream) Seek(index int) {
	if index <= is.index {
		is.index = index // just jump don't update stream state (line,...)
		return
	}
	// seek forward
	is.index = intMin(index, is.size)
}

func (is *InputStream) GetText(start int, stop int) string {
	if stop >= is.size {
		stop = is.size - 1
	}
	if start >= is.size {
		return ""
	}

	return string(is.data[start : stop+1])
}

func (is *InputStream) GetTextFromTokens(start, stop Token) string {
	if start != nil && stop != nil {
		return is.GetTextFromInterval(NewInterval(start.GetTokenIndex(), stop.GetTokenIndex()))
	}

	return ""
}

func (is *InputStream) GetTextFromInterval(i *Interval) string {
	return is.GetText(i.Start, i.Stop)
}

func (*InputStream) GetSourceName() string {
	return "Obtained from string"
}

func (is *InputStream) String() string {
	return string(is.data)
}
