// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"bufio"
	"io"
)

type InputStream struct {
	name  string
	index int
	data  []rune
	size  int
}

// NewIoStream creates a new input stream from the given io.Reader reader.
// Note that the reader is read completely into memory and so it must actually
// have a stopping point - you cannot pass in a reader on an open-ended source such
// as a socket for instance.
func NewIoStream(reader io.Reader) *InputStream {

	rReader := bufio.NewReader(reader)

	is := &InputStream{
		name:  "<empty>",
		index: 0,
	}

	// Pre-build the buffer and read runes reasonably efficiently given that
	// we don't exactly know how big the input is.
	//
	is.data = make([]rune, 0, 512)
	for {
		r, _, err := rReader.ReadRune()
		if err != nil {
			break
		}
		is.data = append(is.data, r)
	}
	is.size = len(is.data) // number of runes
	return is
}

// NewInputStream creates a new input stream from the given string
func NewInputStream(data string) *InputStream {

	is := &InputStream{
		name:  "<empty>",
		index: 0,
		data:  []rune(data), // This is actually the most efficient way
	}
	is.size = len(is.data) // number of runes, but we could also use len(data), which is efficient too
	return is
}

func (is *InputStream) reset() {
	is.index = 0
}

// Consume moves the input pointer to the next character in the input stream
func (is *InputStream) Consume() {
	if is.index >= is.size {
		// assert is.LA(1) == TokenEOF
		panic("cannot consume EOF")
	}
	is.index++
}

// LA returns the character at the given offset from the start of the input stream
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

// LT returns the character at the given offset from the start of the input stream
func (is *InputStream) LT(offset int) int {
	return is.LA(offset)
}

// Index returns the current offset in to the input stream
func (is *InputStream) Index() int {
	return is.index
}

// Size returns the total number of characters in the input stream
func (is *InputStream) Size() int {
	return is.size
}

// Mark does nothing here as we have entire buffer
func (is *InputStream) Mark() int {
	return -1
}

// Release does nothing here as we have entire buffer
func (is *InputStream) Release(_ int) {
}

// Seek the input point to the provided index offset
func (is *InputStream) Seek(index int) {
	if index <= is.index {
		is.index = index // just jump don't update stream state (line,...)
		return
	}
	// seek forward
	is.index = intMin(index, is.size)
}

// GetText returns the text from the input stream from the start to the stop index
func (is *InputStream) GetText(start int, stop int) string {
	if stop >= is.size {
		stop = is.size - 1
	}
	if start >= is.size {
		return ""
	}

	return string(is.data[start : stop+1])
}

// GetTextFromTokens returns the text from the input stream from the first character of the start token to the last
// character of the stop token
func (is *InputStream) GetTextFromTokens(start, stop Token) string {
	if start != nil && stop != nil {
		return is.GetTextFromInterval(NewInterval(start.GetTokenIndex(), stop.GetTokenIndex()))
	}

	return ""
}

func (is *InputStream) GetTextFromInterval(i Interval) string {
	return is.GetText(i.Start, i.Stop)
}

func (*InputStream) GetSourceName() string {
	return "Obtained from string"
}

// String returns the entire input stream as a string
func (is *InputStream) String() string {
	return string(is.data)
}
