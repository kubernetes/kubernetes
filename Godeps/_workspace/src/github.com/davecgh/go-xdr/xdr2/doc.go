/*
 * Copyright (c) 2012-2014 Dave Collins <dave@davec.name>
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
Package xdr implements the data representation portion of the External Data
Representation (XDR) standard protocol as specified in RFC 4506 (obsoletes
RFC 1832 and RFC 1014).

The XDR RFC defines both a data specification language and a data
representation standard.  This package implements methods to encode and decode
XDR data per the data representation standard with the exception of 128-bit
quadruple-precision floating points.  It does not currently implement parsing of
the data specification language.  In other words, the ability to automatically
generate Go code by parsing an XDR data specification file (typically .x
extension) is not supported.  In practice, this limitation of the package is
fairly minor since it is largely unnecessary due to the reflection capabilities
of Go as described below.

This package provides two approaches for encoding and decoding XDR data:

	1) Marshal/Unmarshal functions which automatically map between XDR and Go types
	2) Individual Encoder/Decoder objects to manually work with XDR primitives

For the Marshal/Unmarshal functions, Go reflection capabilities are used to
choose the type of the underlying XDR data based upon the Go type to encode or
the target Go type to decode into.  A description of how each type is mapped is
provided below, however one important type worth reviewing is Go structs.  In
the case of structs, each exported field (first letter capitalized) is reflected
and mapped in order.  As a result, this means a Go struct with exported fields
of the appropriate types listed in the expected order can be used to
automatically encode / decode the XDR data thereby eliminating the need to write
a lot of boilerplate code to encode/decode and error check each piece of XDR
data as is typically required with C based XDR libraries.

Go Type to XDR Type Mappings

The following chart shows an overview of how Go types are mapped to XDR types
for automatic marshalling and unmarshalling.  The documentation for the Marshal
and Unmarshal functions has specific details of how the mapping proceeds.

	Go Type <-> XDR Type
	--------------------
	int8, int16, int32, int <-> XDR Integer
	uint8, uint16, uint32, uint <-> XDR Unsigned Integer
	int64 <-> XDR Hyper Integer
	uint64 <-> XDR Unsigned Hyper Integer
	bool <-> XDR Boolean
	float32 <-> XDR Floating-Point
	float64 <-> XDR Double-Precision Floating-Point
	string <-> XDR String
	byte <-> XDR Integer
	[]byte <-> XDR Variable-Length Opaque Data
	[#]byte <-> XDR Fixed-Length Opaque Data
	[]<type> <-> XDR Variable-Length Array
	[#]<type> <-> XDR Fixed-Length Array
	struct <-> XDR Structure
	map <-> XDR Variable-Length Array of two-element XDR Structures
	time.Time <-> XDR String encoded with RFC3339 nanosecond precision

Notes and Limitations:

	* Automatic marshalling and unmarshalling of variable and fixed-length
	  arrays of uint8s require a special struct tag `xdropaque:"false"`
	  since byte slices and byte arrays are assumed to be opaque data and
	  byte is a Go alias for uint8 thus indistinguishable under reflection
	* Channel, complex, and function types cannot be encoded
	* Interfaces without a concrete value cannot be encoded
	* Cyclic data structures are not supported and will result in infinite
	  loops
	* Strings are marshalled and unmarshalled with UTF-8 character encoding
	  which differs from the XDR specification of ASCII, however UTF-8 is
	  backwards compatible with ASCII so this should rarely cause issues


Encoding

To encode XDR data, use the Marshal function.
	func Marshal(w io.Writer, v interface{}) (int, error)

For example, given the following code snippet:

	type ImageHeader struct {
		Signature	[3]byte
		Version		uint32
		IsGrayscale	bool
		NumSections	uint32
	}
	h := ImageHeader{[3]byte{0xAB, 0xCD, 0xEF}, 2, true, 10}

	var w bytes.Buffer
	bytesWritten, err := xdr.Marshal(&w, &h)
	// Error check elided

The result, encodedData, will then contain the following XDR encoded byte
sequence:

	0xAB, 0xCD, 0xEF, 0x00,
	0x00, 0x00, 0x00, 0x02,
	0x00, 0x00, 0x00, 0x01,
	0x00, 0x00, 0x00, 0x0A


In addition, while the automatic marshalling discussed above will work for the
vast majority of cases, an Encoder object is provided that can be used to
manually encode XDR primitives for complex scenarios where automatic
reflection-based encoding won't work.  The included examples provide a sample of
manual usage via an Encoder.


Decoding

To decode XDR data, use the Unmarshal function.
	func Unmarshal(r io.Reader, v interface{}) (int, error)

For example, given the following code snippet:

	type ImageHeader struct {
		Signature	[3]byte
		Version		uint32
		IsGrayscale	bool
		NumSections	uint32
	}

	// Using output from the Encoding section above.
	encodedData := []byte{
		0xAB, 0xCD, 0xEF, 0x00,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x0A,
	}

	var h ImageHeader
	bytesRead, err := xdr.Unmarshal(bytes.NewReader(encodedData), &h)
	// Error check elided

The struct instance, h, will then contain the following values:

	h.Signature = [3]byte{0xAB, 0xCD, 0xEF}
	h.Version = 2
	h.IsGrayscale = true
	h.NumSections = 10

In addition, while the automatic unmarshalling discussed above will work for the
vast majority of cases, a Decoder object is provided that can be used to
manually decode XDR primitives for complex scenarios where automatic
reflection-based decoding won't work.  The included examples provide a sample of
manual usage via a Decoder.

Errors

All errors are either of type UnmarshalError or MarshalError.  Both provide
human-readable output as well as an ErrorCode field which can be inspected by
sophisticated callers if necessary.

See the documentation of UnmarshalError, MarshalError, and ErrorCode for further
details.
*/
package xdr
