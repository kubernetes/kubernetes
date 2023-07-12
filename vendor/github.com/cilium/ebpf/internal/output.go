package internal

import (
	"bytes"
	"errors"
	"go/format"
	"go/scanner"
	"io"
	"reflect"
	"strings"
	"unicode"
)

// Identifier turns a C style type or field name into an exportable Go equivalent.
func Identifier(str string) string {
	prev := rune(-1)
	return strings.Map(func(r rune) rune {
		// See https://golang.org/ref/spec#Identifiers
		switch {
		case unicode.IsLetter(r):
			if prev == -1 {
				r = unicode.ToUpper(r)
			}

		case r == '_':
			switch {
			// The previous rune was deleted, or we are at the
			// beginning of the string.
			case prev == -1:
				fallthrough

			// The previous rune is a lower case letter or a digit.
			case unicode.IsDigit(prev) || (unicode.IsLetter(prev) && unicode.IsLower(prev)):
				// delete the current rune, and force the
				// next character to be uppercased.
				r = -1
			}

		case unicode.IsDigit(r):

		default:
			// Delete the current rune. prev is unchanged.
			return -1
		}

		prev = r
		return r
	}, str)
}

// WriteFormatted outputs a formatted src into out.
//
// If formatting fails it returns an informative error message.
func WriteFormatted(src []byte, out io.Writer) error {
	formatted, err := format.Source(src)
	if err == nil {
		_, err = out.Write(formatted)
		return err
	}

	var el scanner.ErrorList
	if !errors.As(err, &el) {
		return err
	}

	var nel scanner.ErrorList
	for _, err := range el {
		if !err.Pos.IsValid() {
			nel = append(nel, err)
			continue
		}

		buf := src[err.Pos.Offset:]
		nl := bytes.IndexRune(buf, '\n')
		if nl == -1 {
			nel = append(nel, err)
			continue
		}

		err.Msg += ": " + string(buf[:nl])
		nel = append(nel, err)
	}

	return nel
}

// GoTypeName is like %T, but elides the package name.
//
// Pointers to a type are peeled off.
func GoTypeName(t any) string {
	rT := reflect.TypeOf(t)
	for rT.Kind() == reflect.Pointer {
		rT = rT.Elem()
	}
	// Doesn't return the correct Name for generic types due to https://github.com/golang/go/issues/55924
	return rT.Name()
}
