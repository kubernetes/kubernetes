/*
 * Copyright (c) 2013 Dave Collins <dave@davec.name>
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

/*
Package spew implements a deep pretty printer for Go data structures to aid in
debugging.

A quick overview of the additional features spew provides over the built-in
printing facilities for Go data types are as follows:

	* Pointers are dereferenced and followed
	* Circular data structures are detected and handled properly
	* Custom Stringer/error interfaces are optionally invoked, including
	  on unexported types
	* Custom types which only implement the Stringer/error interfaces via
	  a pointer receiver are optionally invoked when passing non-pointer
	  variables
	* Byte arrays and slices are dumped like the hexdump -C command which
	  includes offsets, byte values in hex, and ASCII output (only when using
	  Dump style)

There are two different approaches spew allows for dumping Go data structures:

	* Dump style which prints with newlines, customizable indentation,
	  and additional debug information such as types and all pointer addresses
	  used to indirect to the final value
	* A custom Formatter interface that integrates cleanly with the standard fmt
	  package and replaces %v, %+v, %#v, and %#+v to provide inline printing
	  similar to the default %v while providing the additional functionality
	  outlined above and passing unsupported format verbs such as %x and %q
	  along to fmt

Quick Start

This section demonstrates how to quickly get started with spew.  See the
sections below for further details on formatting and configuration options.

To dump a variable with full newlines, indentation, type, and pointer
information use Dump, Fdump, or Sdump:
	spew.Dump(myVar1, myVar2, ...)
	spew.Fdump(someWriter, myVar1, myVar2, ...)
	str := spew.Sdump(myVar1, myVar2, ...)

Alternatively, if you would prefer to use format strings with a compacted inline
printing style, use the convenience wrappers Printf, Fprintf, etc with
%v (most compact), %+v (adds pointer addresses), %#v (adds types), or
%#+v (adds types and pointer addresses):
	spew.Printf("myVar1: %v -- myVar2: %+v", myVar1, myVar2)
	spew.Printf("myVar3: %#v -- myVar4: %#+v", myVar3, myVar4)
	spew.Fprintf(someWriter, "myVar1: %v -- myVar2: %+v", myVar1, myVar2)
	spew.Fprintf(someWriter, "myVar3: %#v -- myVar4: %#+v", myVar3, myVar4)

Configuration Options

Configuration of spew is handled by fields in the ConfigState type.  For
convenience, all of the top-level functions use a global state available
via the spew.Config global.

It is also possible to create a ConfigState instance that provides methods
equivalent to the top-level functions.  This allows concurrent configuration
options.  See the ConfigState documentation for more details.

The following configuration options are available:
	* Indent
		String to use for each indentation level for Dump functions.
		It is a single space by default.  A popular alternative is "\t".

	* MaxDepth
		Maximum number of levels to descend into nested data structures.
		There is no limit by default.

	* DisableMethods
		Disables invocation of error and Stringer interface methods.
		Method invocation is enabled by default.

	* DisablePointerMethods
		Disables invocation of error and Stringer interface methods on types
		which only accept pointer receivers from non-pointer variables.
		Pointer method invocation is enabled by default.

	* ContinueOnMethod
		Enables recursion into types after invoking error and Stringer interface
		methods. Recursion after method invocation is disabled by default.

	* SortKeys
		Specifies map keys should be sorted before being printed. Use
		this to have a more deterministic, diffable output.  Note that
		only native types (bool, int, uint, floats, uintptr and string)
		and types which implement error or Stringer interfaces are
		supported with other types sorted according to the
		reflect.Value.String() output which guarantees display
		stability.  Natural map order is used by default.

	* SpewKeys
		Specifies that, as a last resort attempt, map keys should be
		spewed to strings and sorted by those strings.  This is only
		considered if SortKeys is true.

Dump Usage

Simply call spew.Dump with a list of variables you want to dump:

	spew.Dump(myVar1, myVar2, ...)

You may also call spew.Fdump if you would prefer to output to an arbitrary
io.Writer.  For example, to dump to standard error:

	spew.Fdump(os.Stderr, myVar1, myVar2, ...)

A third option is to call spew.Sdump to get the formatted output as a string:

	str := spew.Sdump(myVar1, myVar2, ...)

Sample Dump Output

See the Dump example for details on the setup of the types and variables being
shown here.

	(main.Foo) {
	 unexportedField: (*main.Bar)(0xf84002e210)({
	  flag: (main.Flag) flagTwo,
	  data: (uintptr) <nil>
	 }),
	 ExportedField: (map[interface {}]interface {}) (len=1) {
	  (string) (len=3) "one": (bool) true
	 }
	}

Byte (and uint8) arrays and slices are displayed uniquely like the hexdump -C
command as shown.
	([]uint8) (len=32 cap=32) {
	 00000000  11 12 13 14 15 16 17 18  19 1a 1b 1c 1d 1e 1f 20  |............... |
	 00000010  21 22 23 24 25 26 27 28  29 2a 2b 2c 2d 2e 2f 30  |!"#$%&'()*+,-./0|
	 00000020  31 32                                             |12|
	}

Custom Formatter

Spew provides a custom formatter that implements the fmt.Formatter interface
so that it integrates cleanly with standard fmt package printing functions. The
formatter is useful for inline printing of smaller data types similar to the
standard %v format specifier.

The custom formatter only responds to the %v (most compact), %+v (adds pointer
addresses), %#v (adds types), or %#+v (adds types and pointer addresses) verb
combinations.  Any other verbs such as %x and %q will be sent to the the
standard fmt package for formatting.  In addition, the custom formatter ignores
the width and precision arguments (however they will still work on the format
specifiers not handled by the custom formatter).

Custom Formatter Usage

The simplest way to make use of the spew custom formatter is to call one of the
convenience functions such as spew.Printf, spew.Println, or spew.Printf.  The
functions have syntax you are most likely already familiar with:

	spew.Printf("myVar1: %v -- myVar2: %+v", myVar1, myVar2)
	spew.Printf("myVar3: %#v -- myVar4: %#+v", myVar3, myVar4)
	spew.Println(myVar, myVar2)
	spew.Fprintf(os.Stderr, "myVar1: %v -- myVar2: %+v", myVar1, myVar2)
	spew.Fprintf(os.Stderr, "myVar3: %#v -- myVar4: %#+v", myVar3, myVar4)

See the Index for the full list convenience functions.

Sample Formatter Output

Double pointer to a uint8:
	  %v: <**>5
	 %+v: <**>(0xf8400420d0->0xf8400420c8)5
	 %#v: (**uint8)5
	%#+v: (**uint8)(0xf8400420d0->0xf8400420c8)5

Pointer to circular struct with a uint8 field and a pointer to itself:
	  %v: <*>{1 <*><shown>}
	 %+v: <*>(0xf84003e260){ui8:1 c:<*>(0xf84003e260)<shown>}
	 %#v: (*main.circular){ui8:(uint8)1 c:(*main.circular)<shown>}
	%#+v: (*main.circular)(0xf84003e260){ui8:(uint8)1 c:(*main.circular)(0xf84003e260)<shown>}

See the Printf example for details on the setup of variables being shown
here.

Errors

Since it is possible for custom Stringer/error interfaces to panic, spew
detects them and handles them internally by printing the panic information
inline with the output.  Since spew is intended to provide deep pretty printing
capabilities on structures, it intentionally does not return any errors.
*/
package spew
