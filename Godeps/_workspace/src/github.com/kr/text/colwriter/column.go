// Package colwriter provides a write filter that formats
// input lines in multiple columns.
//
// The package is a straightforward translation from
// /src/cmd/draw/mc.c in Plan 9 from User Space.
package colwriter

import (
	"bytes"
	"io"
	"unicode/utf8"
)

const (
	tab = 4
)

const (
	// Print each input line ending in a colon ':' separately.
	BreakOnColon uint = 1 << iota
)

// A Writer is a filter that arranges input lines in as many columns as will
// fit in its width. Tab '\t' chars in the input are translated to sequences
// of spaces ending at multiples of 4 positions.
//
// If BreakOnColon is set, each input line ending in a colon ':' is written
// separately.
//
// The Writer assumes that all Unicode code points have the same width; this
// may not be true in some fonts.
type Writer struct {
	w     io.Writer
	buf   []byte
	width int
	flag  uint
}

// NewWriter allocates and initializes a new Writer writing to w.
// Parameter width controls the total number of characters on each line
// across all columns.
func NewWriter(w io.Writer, width int, flag uint) *Writer {
	return &Writer{
		w:     w,
		width: width,
		flag:  flag,
	}
}

// Write writes p to the writer w. The only errors returned are ones
// encountered while writing to the underlying output stream.
func (w *Writer) Write(p []byte) (n int, err error) {
	var linelen int
	var lastWasColon bool
	for i, c := range p {
		w.buf = append(w.buf, c)
		linelen++
		if c == '\t' {
			w.buf[len(w.buf)-1] = ' '
			for linelen%tab != 0 {
				w.buf = append(w.buf, ' ')
				linelen++
			}
		}
		if w.flag&BreakOnColon != 0 && c == ':' {
			lastWasColon = true
		} else if lastWasColon {
			if c == '\n' {
				pos := bytes.LastIndex(w.buf[:len(w.buf)-1], []byte{'\n'})
				if pos < 0 {
					pos = 0
				}
				line := w.buf[pos:]
				w.buf = w.buf[:pos]
				if err = w.columnate(); err != nil {
					if len(line) < i {
						return i - len(line), err
					}
					return 0, err
				}
				if n, err := w.w.Write(line); err != nil {
					if r := len(line) - n; r < i {
						return i - r, err
					}
					return 0, err
				}
			}
			lastWasColon = false
		}
		if c == '\n' {
			linelen = 0
		}
	}
	return len(p), nil
}

// Flush should be called after the last call to Write to ensure that any data
// buffered in the Writer is written to output.
func (w *Writer) Flush() error {
	return w.columnate()
}

func (w *Writer) columnate() error {
	words := bytes.Split(w.buf, []byte{'\n'})
	w.buf = nil
	if len(words[len(words)-1]) == 0 {
		words = words[:len(words)-1]
	}
	maxwidth := 0
	for _, wd := range words {
		if n := utf8.RuneCount(wd); n > maxwidth {
			maxwidth = n
		}
	}
	maxwidth++ // space char
	wordsPerLine := w.width / maxwidth
	if wordsPerLine <= 0 {
		wordsPerLine = 1
	}
	nlines := (len(words) + wordsPerLine - 1) / wordsPerLine
	for i := 0; i < nlines; i++ {
		col := 0
		endcol := 0
		for j := i; j < len(words); j += nlines {
			endcol += maxwidth
			_, err := w.w.Write(words[j])
			if err != nil {
				return err
			}
			col += utf8.RuneCount(words[j])
			if j+nlines < len(words) {
				for col < endcol {
					_, err := w.w.Write([]byte{' '})
					if err != nil {
						return err
					}
					col++
				}
			}
		}
		_, err := w.w.Write([]byte{'\n'})
		if err != nil {
			return err
		}
	}
	return nil
}
