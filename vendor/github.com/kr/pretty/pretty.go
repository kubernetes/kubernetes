// Package pretty provides pretty-printing for Go values. This is
// useful during debugging, to avoid wrapping long output lines in
// the terminal.
//
// It provides a function, Formatter, that can be used with any
// function that accepts a format string. It also provides
// convenience wrappers for functions in packages fmt and log.
package pretty

import (
	"fmt"
	"io"
	"log"
	"reflect"
)

// Errorf is a convenience wrapper for fmt.Errorf.
//
// Calling Errorf(f, x, y) is equivalent to
// fmt.Errorf(f, Formatter(x), Formatter(y)).
func Errorf(format string, a ...interface{}) error {
	return fmt.Errorf(format, wrap(a, false)...)
}

// Fprintf is a convenience wrapper for fmt.Fprintf.
//
// Calling Fprintf(w, f, x, y) is equivalent to
// fmt.Fprintf(w, f, Formatter(x), Formatter(y)).
func Fprintf(w io.Writer, format string, a ...interface{}) (n int, error error) {
	return fmt.Fprintf(w, format, wrap(a, false)...)
}

// Log is a convenience wrapper for log.Printf.
//
// Calling Log(x, y) is equivalent to
// log.Print(Formatter(x), Formatter(y)), but each operand is
// formatted with "%# v".
func Log(a ...interface{}) {
	log.Print(wrap(a, true)...)
}

// Logf is a convenience wrapper for log.Printf.
//
// Calling Logf(f, x, y) is equivalent to
// log.Printf(f, Formatter(x), Formatter(y)).
func Logf(format string, a ...interface{}) {
	log.Printf(format, wrap(a, false)...)
}

// Logln is a convenience wrapper for log.Printf.
//
// Calling Logln(x, y) is equivalent to
// log.Println(Formatter(x), Formatter(y)), but each operand is
// formatted with "%# v".
func Logln(a ...interface{}) {
	log.Println(wrap(a, true)...)
}

// Print pretty-prints its operands and writes to standard output.
//
// Calling Print(x, y) is equivalent to
// fmt.Print(Formatter(x), Formatter(y)), but each operand is
// formatted with "%# v".
func Print(a ...interface{}) (n int, errno error) {
	return fmt.Print(wrap(a, true)...)
}

// Printf is a convenience wrapper for fmt.Printf.
//
// Calling Printf(f, x, y) is equivalent to
// fmt.Printf(f, Formatter(x), Formatter(y)).
func Printf(format string, a ...interface{}) (n int, errno error) {
	return fmt.Printf(format, wrap(a, false)...)
}

// Println pretty-prints its operands and writes to standard output.
//
// Calling Print(x, y) is equivalent to
// fmt.Println(Formatter(x), Formatter(y)), but each operand is
// formatted with "%# v".
func Println(a ...interface{}) (n int, errno error) {
	return fmt.Println(wrap(a, true)...)
}

// Sprint is a convenience wrapper for fmt.Sprintf.
//
// Calling Sprint(x, y) is equivalent to
// fmt.Sprint(Formatter(x), Formatter(y)), but each operand is
// formatted with "%# v".
func Sprint(a ...interface{}) string {
	return fmt.Sprint(wrap(a, true)...)
}

// Sprintf is a convenience wrapper for fmt.Sprintf.
//
// Calling Sprintf(f, x, y) is equivalent to
// fmt.Sprintf(f, Formatter(x), Formatter(y)).
func Sprintf(format string, a ...interface{}) string {
	return fmt.Sprintf(format, wrap(a, false)...)
}

func wrap(a []interface{}, force bool) []interface{} {
	w := make([]interface{}, len(a))
	for i, x := range a {
		w[i] = formatter{v: reflect.ValueOf(x), force: force}
	}
	return w
}
