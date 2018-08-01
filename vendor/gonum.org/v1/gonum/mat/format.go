// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"strconv"
)

// Formatted returns a fmt.Formatter for the matrix m using the given options.
func Formatted(m Matrix, options ...FormatOption) fmt.Formatter {
	f := formatter{
		matrix: m,
		dot:    '.',
	}
	for _, o := range options {
		o(&f)
	}
	return f
}

type formatter struct {
	matrix  Matrix
	prefix  string
	margin  int
	dot     byte
	squeeze bool
}

// FormatOption is a functional option for matrix formatting.
type FormatOption func(*formatter)

// Prefix sets the formatted prefix to the string p. Prefix is a string that is prepended to
// each line of output.
func Prefix(p string) FormatOption {
	return func(f *formatter) { f.prefix = p }
}

// Excerpt sets the maximum number of rows and columns to print at the margins of the matrix
// to m. If m is zero or less all elements are printed.
func Excerpt(m int) FormatOption {
	return func(f *formatter) { f.margin = m }
}

// DotByte sets the dot character to b. The dot character is used to replace zero elements
// if the result is printed with the fmt ' ' verb flag. Without a DotByte option, the default
// dot character is '.'.
func DotByte(b byte) FormatOption {
	return func(f *formatter) { f.dot = b }
}

// Squeeze sets the printing behaviour to minimise column width for each individual column.
func Squeeze() FormatOption {
	return func(f *formatter) { f.squeeze = true }
}

// Format satisfies the fmt.Formatter interface.
func (f formatter) Format(fs fmt.State, c rune) {
	if c == 'v' && fs.Flag('#') {
		fmt.Fprintf(fs, "%#v", f.matrix)
		return
	}
	format(f.matrix, f.prefix, f.margin, f.dot, f.squeeze, fs, c)
}

// format prints a pretty representation of m to the fs io.Writer. The format character c
// specifies the numerical representation of of elements; valid values are those for float64
// specified in the fmt package, with their associated flags. In addition to this, a space
// preceding a verb indicates that zero values should be represented by the dot character.
// The printed range of the matrix can be limited by specifying a positive value for margin;
// If margin is greater than zero, only the first and last margin rows/columns of the matrix
// are output. If squeeze is true, column widths are determined on a per-column basis.
//
// format will not provide Go syntax output.
func format(m Matrix, prefix string, margin int, dot byte, squeeze bool, fs fmt.State, c rune) {
	rows, cols := m.Dims()

	var printed int
	if margin <= 0 {
		printed = rows
		if cols > printed {
			printed = cols
		}
	} else {
		printed = margin
	}

	prec, pOk := fs.Precision()
	if !pOk {
		prec = -1
	}

	var (
		maxWidth int
		widths   widther
		buf, pad []byte
	)
	if squeeze {
		widths = make(columnWidth, cols)
	} else {
		widths = new(uniformWidth)
	}
	switch c {
	case 'v', 'e', 'E', 'f', 'F', 'g', 'G':
		if c == 'v' {
			buf, maxWidth = maxCellWidth(m, 'g', printed, prec, widths)
		} else {
			buf, maxWidth = maxCellWidth(m, c, printed, prec, widths)
		}
	default:
		fmt.Fprintf(fs, "%%!%c(%T=Dims(%d, %d))", c, m, rows, cols)
		return
	}
	width, _ := fs.Width()
	width = max(width, maxWidth)
	pad = make([]byte, max(width, 2))
	for i := range pad {
		pad[i] = ' '
	}

	first := true
	if rows > 2*printed || cols > 2*printed {
		first = false
		fmt.Fprintf(fs, "Dims(%d, %d)\n", rows, cols)
	}

	skipZero := fs.Flag(' ')
	for i := 0; i < rows; i++ {
		if !first {
			fmt.Fprint(fs, prefix)
		}
		first = false
		var el string
		switch {
		case rows == 1:
			fmt.Fprint(fs, "[")
			el = "]"
		case i == 0:
			fmt.Fprint(fs, "⎡")
			el = "⎤\n"
		case i < rows-1:
			fmt.Fprint(fs, "⎢")
			el = "⎥\n"
		default:
			fmt.Fprint(fs, "⎣")
			el = "⎦"
		}

		for j := 0; j < cols; j++ {
			if j >= printed && j < cols-printed {
				j = cols - printed - 1
				if i == 0 || i == rows-1 {
					fmt.Fprint(fs, "...  ...  ")
				} else {
					fmt.Fprint(fs, "          ")
				}
				continue
			}

			v := m.At(i, j)
			if v == 0 && skipZero {
				buf = buf[:1]
				buf[0] = dot
			} else {
				if c == 'v' {
					buf = strconv.AppendFloat(buf[:0], v, 'g', prec, 64)
				} else {
					buf = strconv.AppendFloat(buf[:0], v, byte(c), prec, 64)
				}
			}
			if fs.Flag('-') {
				fs.Write(buf)
				fs.Write(pad[:widths.width(j)-len(buf)])
			} else {
				fs.Write(pad[:widths.width(j)-len(buf)])
				fs.Write(buf)
			}

			if j < cols-1 {
				fs.Write(pad[:2])
			}
		}

		fmt.Fprint(fs, el)

		if i >= printed-1 && i < rows-printed && 2*printed < rows {
			i = rows - printed - 1
			fmt.Fprintf(fs, "%s .\n%[1]s .\n%[1]s .\n", prefix)
			continue
		}
	}
}

func maxCellWidth(m Matrix, c rune, printed, prec int, w widther) ([]byte, int) {
	var (
		buf        = make([]byte, 0, 64)
		rows, cols = m.Dims()
		max        int
	)
	for i := 0; i < rows; i++ {
		if i >= printed-1 && i < rows-printed && 2*printed < rows {
			i = rows - printed - 1
			continue
		}
		for j := 0; j < cols; j++ {
			if j >= printed && j < cols-printed {
				continue
			}

			buf = strconv.AppendFloat(buf, m.At(i, j), byte(c), prec, 64)
			if len(buf) > max {
				max = len(buf)
			}
			if len(buf) > w.width(j) {
				w.setWidth(j, len(buf))
			}
			buf = buf[:0]
		}
	}
	return buf, max
}

type widther interface {
	width(i int) int
	setWidth(i, w int)
}

type uniformWidth int

func (u *uniformWidth) width(_ int) int   { return int(*u) }
func (u *uniformWidth) setWidth(_, w int) { *u = uniformWidth(w) }

type columnWidth []int

func (c columnWidth) width(i int) int   { return c[i] }
func (c columnWidth) setWidth(i, w int) { c[i] = w }
