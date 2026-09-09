/*
 * Copyright (c) 2013-2016 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package spew

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

// ConfigState houses the configuration options used by spew to format and
// display values.  There is a global instance, Config, that is used to control
// all top-level Formatter and Dump functionality.  Each ConfigState instance
// provides methods equivalent to the top-level functions.
//
// The zero value for ConfigState provides no indentation.  You would typically
// want to set it to a space or a tab.
//
// Alternatively, you can use NewDefaultConfig to get a ConfigState instance
// with default settings.  See the documentation of NewDefaultConfig for default
// values.
type ConfigState struct {
	// Indent specifies the string to use for each indentation level.  The
	// global config instance that all top-level functions use set this to a
	// single space by default.  If you would like more indentation, you might
	// set this to a tab with "\t" or perhaps two spaces with "  ".
	Indent string

	// MaxDepth controls the maximum number of levels to descend into nested
	// data structures.  The default, 0, means there is no limit.
	//
	// NOTE: Circular data structures are properly detected, so it is not
	// necessary to set this value unless you specifically want to limit deeply
	// nested data structures.
	MaxDepth int

	// DisableMethods specifies whether or not error and Stringer interfaces are
	// invoked for types that implement them.
	DisableMethods bool

	// DisablePointerMethods specifies whether or not to check for and invoke
	// error and Stringer interfaces on types which only accept a pointer
	// receiver when the current type is not a pointer.
	//
	// NOTE: This might be an unsafe action since calling one of these methods
	// with a pointer receiver could technically mutate the value, however,
	// in practice, types which choose to satisify an error or Stringer
	// interface with a pointer receiver should not be mutating their state
	// inside these interface methods.  As a result, this option relies on
	// access to the unsafe package, so it will not have any effect when
	// running in environments without access to the unsafe package such as
	// Google App Engine or with the "safe" build tag specified.
	DisablePointerMethods bool

	// DisablePointerAddresses specifies whether to disable the printing of
	// pointer addresses. This is useful when diffing data structures in tests.
	DisablePointerAddresses bool

	// DisableCapacities specifies whether to disable the printing of capacities
	// for arrays, slices, maps and channels. This is useful when diffing
	// data structures in tests.
	DisableCapacities bool

	// ContinueOnMethod specifies whether or not recursion should continue once
	// a custom error or Stringer interface is invoked.  The default, false,
	// means it will print the results of invoking the custom error or Stringer
	// interface and return immediately instead of continuing to recurse into
	// the internals of the data type.
	//
	// NOTE: This flag does not have any effect if method invocation is disabled
	// via the DisableMethods or DisablePointerMethods options.
	ContinueOnMethod bool

	// SortKeys specifies map keys should be sorted before being printed. Use
	// this to have a more deterministic, diffable output.  Note that only
	// native types (bool, int, uint, floats, uintptr and string) and types
	// that support the error or Stringer interfaces (if methods are
	// enabled) are supported, with other types sorted according to the
	// reflect.Value.String() output which guarantees display stability.
	SortKeys bool

	// SpewKeys specifies that, as a last resort attempt, map keys should
	// be spewed to strings and sorted by those strings.  This is only
	// considered if SortKeys is true.
	SpewKeys bool
}

// Config is the active configuration of the top-level functions.
// The configuration can be changed by modifying the contents of spew.Config.
var Config = ConfigState{Indent: " "}

// Errorf is a wrapper for fmt.Errorf that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the formatted string as a value that satisfies error.  See NewFormatter
// for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Errorf(format, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Errorf(format string, a ...interface{}) (err error) {
	return fmt.Errorf(format, c.convertArgs(a)...)
}

// Fprint is a wrapper for fmt.Fprint that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the number of bytes written and any write error encountered.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Fprint(w, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Fprint(w io.Writer, a ...interface{}) (n int, err error) {
	return fmt.Fprint(w, c.convertArgs(a)...)
}

// Fprintf is a wrapper for fmt.Fprintf that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the number of bytes written and any write error encountered.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Fprintf(w, format, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Fprintf(w io.Writer, format string, a ...interface{}) (n int, err error) {
	return fmt.Fprintf(w, format, c.convertArgs(a)...)
}

// Fprintln is a wrapper for fmt.Fprintln that treats each argument as if it
// passed with a Formatter interface returned by c.NewFormatter.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Fprintln(w, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Fprintln(w io.Writer, a ...interface{}) (n int, err error) {
	return fmt.Fprintln(w, c.convertArgs(a)...)
}

// Print is a wrapper for fmt.Print that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the number of bytes written and any write error encountered.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Print(c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Print(a ...interface{}) (n int, err error) {
	return fmt.Print(c.convertArgs(a)...)
}

// Printf is a wrapper for fmt.Printf that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the number of bytes written and any write error encountered.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Printf(format, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Printf(format string, a ...interface{}) (n int, err error) {
	return fmt.Printf(format, c.convertArgs(a)...)
}

// Println is a wrapper for fmt.Println that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the number of bytes written and any write error encountered.  See
// NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Println(c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Println(a ...interface{}) (n int, err error) {
	return fmt.Println(c.convertArgs(a)...)
}

// Sprint is a wrapper for fmt.Sprint that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the resulting string.  See NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Sprint(c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Sprint(a ...interface{}) string {
	return fmt.Sprint(c.convertArgs(a)...)
}

// Sprintf is a wrapper for fmt.Sprintf that treats each argument as if it were
// passed with a Formatter interface returned by c.NewFormatter.  It returns
// the resulting string.  See NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Sprintf(format, c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Sprintf(format string, a ...interface{}) string {
	return fmt.Sprintf(format, c.convertArgs(a)...)
}

// Sprintln is a wrapper for fmt.Sprintln that treats each argument as if it
// were passed with a Formatter interface returned by c.NewFormatter.  It
// returns the resulting string.  See NewFormatter for formatting details.
//
// This function is shorthand for the following syntax:
//
//	fmt.Sprintln(c.NewFormatter(a), c.NewFormatter(b))
func (c *ConfigState) Sprintln(a ...interface{}) string {
	return fmt.Sprintln(c.convertArgs(a)...)
}

/*
NewFormatter returns a custom formatter that satisfies the fmt.Formatter
interface.  As a result, it integrates cleanly with standard fmt package
printing functions.  The formatter is useful for inline printing of smaller data
types similar to the standard %v format specifier.

The custom formatter only responds to the %v (most compact), %+v (adds pointer
addresses), %#v (adds types), and %#+v (adds types and pointer addresses) verb
combinations.  Any other verbs such as %x and %q will be sent to the the
standard fmt package for formatting.  In addition, the custom formatter ignores
the width and precision arguments (however they will still work on the format
specifiers not handled by the custom formatter).

Typically this function shouldn't be called directly.  It is much easier to make
use of the custom formatter by calling one of the convenience functions such as
c.Printf, c.Println, or c.Printf.
*/
func (c *ConfigState) NewFormatter(v interface{}) fmt.Formatter {
	return newFormatter(c, v)
}

// Fdump formats and displays the passed arguments to io.Writer w.  It formats
// exactly the same as Dump.
func (c *ConfigState) Fdump(w io.Writer, a ...interface{}) {
	fdump(c, w, a...)
}

/*
Dump displays the passed parameters to standard out with newlines, customizable
indentation, and additional debug information such as complete types and all
pointer addresses used to indirect to the final value.  It provides the
following features over the built-in printing facilities provided by the fmt
package:

  - Pointers are dereferenced and followed
  - Circular data structures are detected and handled properly
  - Custom Stringer/error interfaces are optionally invoked, including
    on unexported types
  - Custom types which only implement the Stringer/error interfaces via
    a pointer receiver are optionally invoked when passing non-pointer
    variables
  - Byte arrays and slices are dumped like the hexdump -C command which
    includes offsets, byte values in hex, and ASCII output

The configuration options are controlled by modifying the public members
of c.  See ConfigState for options documentation.

See Fdump if you would prefer dumping to an arbitrary io.Writer or Sdump to
get the formatted result as a string.
*/
func (c *ConfigState) Dump(a ...interface{}) {
	fdump(c, os.Stdout, a...)
}

// Sdump returns a string with the passed arguments formatted exactly the same
// as Dump.
func (c *ConfigState) Sdump(a ...interface{}) string {
	var buf bytes.Buffer
	fdump(c, &buf, a...)
	return buf.String()
}

// convertArgs accepts a slice of arguments and returns a slice of the same
// length with each argument converted to a spew Formatter interface using
// the ConfigState associated with s.
func (c *ConfigState) convertArgs(args []interface{}) (formatters []interface{}) {
	formatters = make([]interface{}, len(args))
	for index, arg := range args {
		formatters[index] = newFormatter(c, arg)
	}
	return formatters
}

// NewDefaultConfig returns a ConfigState with the following default settings.
//
//	Indent: " "
//	MaxDepth: 0
//	DisableMethods: false
//	DisablePointerMethods: false
//	ContinueOnMethod: false
//	SortKeys: false
func NewDefaultConfig() *ConfigState {
	return &ConfigState{Indent: " "}
}
