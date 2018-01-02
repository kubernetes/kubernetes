// Package exlib contains common library code for the examples.
package exlib

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

var progname = filepath.Base(os.Args[0])

// Before set to 5 minutes; certificates will attempt to auto-update 5
// minutes before they expire.
var Before = 5 * time.Minute

// Err displays a formatting error message to standard error,
// appending the error string, and exits with the status code from
// `exit`, à la err(3).
func Err(exit int, err error, format string, a ...interface{}) {
	format = fmt.Sprintf("[%s] %s", progname, format)
	format += ": %v\n"
	a = append(a, err)
	fmt.Fprintf(os.Stderr, format, a...)
	os.Exit(exit)
}

// Errx displays a formatted error message to standard error and exits
// with the status code from `exit`, à la errx(3).
func Errx(exit int, format string, a ...interface{}) {
	format = fmt.Sprintf("[%s] %s", progname, format)
	format += "\n"
	fmt.Fprintf(os.Stderr, format, a...)
	os.Exit(exit)
}

// Warn displays a formatted error message to standard output,
// appending the error string, à la warn(3).
func Warn(err error, format string, a ...interface{}) (int, error) {
	format = fmt.Sprintf("[%s] %s", progname, format)
	format += ": %v\n"
	a = append(a, err)
	return fmt.Fprintf(os.Stderr, format, a...)
}

// Unpack reads a message from an io.Reader.
func Unpack(r io.Reader) ([]byte, error) {
	var bl [2]byte

	_, err := io.ReadFull(r, bl[:])
	if err != nil {
		return nil, err
	}

	n := binary.LittleEndian.Uint16(bl[:])
	buf := make([]byte, int(n))
	_, err = io.ReadFull(r, buf)
	return buf, err
}

const messageMax = 1 << 16

// Pack writes a message to an io.Writer.
func Pack(w io.Writer, buf []byte) error {
	if len(buf) > messageMax {
		return errors.New("message is too large")
	}

	var bl [2]byte
	binary.LittleEndian.PutUint16(bl[:], uint16(len(buf)))
	_, err := w.Write(bl[:])
	if err != nil {
		return err
	}

	_, err = w.Write(buf)
	return err
}
