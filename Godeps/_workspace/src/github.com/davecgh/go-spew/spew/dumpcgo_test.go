// Copyright (c) 2013 Dave Collins <dave@davec.name>
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

// NOTE: Due to the following build constraints, this file will only be compiled
// when both cgo is supported and "-tags testcgo" is added to the go test
// command line.  This means the cgo tests are only added (and hence run) when
// specifially requested.  This configuration is used because spew itself
// does not require cgo to run even though it does handle certain cgo types
// specially.  Rather than forcing all clients to require cgo and an external
// C compiler just to run the tests, this scheme makes them optional.
// +build cgo,testcgo

package spew_test

import (
	"fmt"
	"github.com/davecgh/go-spew/spew/testdata"
)

func addCgoDumpTests() {
	// C char pointer.
	v := testdata.GetCgoCharPointer()
	nv := testdata.GetCgoNullCharPointer()
	pv := &v
	vcAddr := fmt.Sprintf("%p", v)
	vAddr := fmt.Sprintf("%p", pv)
	pvAddr := fmt.Sprintf("%p", &pv)
	vt := "*testdata._Ctype_char"
	vs := "116"
	addDumpTest(v, "("+vt+")("+vcAddr+")("+vs+")\n")
	addDumpTest(pv, "(*"+vt+")("+vAddr+"->"+vcAddr+")("+vs+")\n")
	addDumpTest(&pv, "(**"+vt+")("+pvAddr+"->"+vAddr+"->"+vcAddr+")("+vs+")\n")
	addDumpTest(nv, "("+vt+")(<nil>)\n")

	// C char array.
	v2, v2l, v2c := testdata.GetCgoCharArray()
	v2Len := fmt.Sprintf("%d", v2l)
	v2Cap := fmt.Sprintf("%d", v2c)
	v2t := "[6]testdata._Ctype_char"
	v2s := "(len=" + v2Len + " cap=" + v2Cap + ") " +
		"{\n 00000000  74 65 73 74 32 00                               " +
		"  |test2.|\n}"
	addDumpTest(v2, "("+v2t+") "+v2s+"\n")

	// C unsigned char array.
	v3, v3l, v3c := testdata.GetCgoUnsignedCharArray()
	v3Len := fmt.Sprintf("%d", v3l)
	v3Cap := fmt.Sprintf("%d", v3c)
	v3t := "[6]testdata._Ctype_unsignedchar"
	v3s := "(len=" + v3Len + " cap=" + v3Cap + ") " +
		"{\n 00000000  74 65 73 74 33 00                               " +
		"  |test3.|\n}"
	addDumpTest(v3, "("+v3t+") "+v3s+"\n")

	// C signed char array.
	v4, v4l, v4c := testdata.GetCgoSignedCharArray()
	v4Len := fmt.Sprintf("%d", v4l)
	v4Cap := fmt.Sprintf("%d", v4c)
	v4t := "[6]testdata._Ctype_schar"
	v4t2 := "testdata._Ctype_schar"
	v4s := "(len=" + v4Len + " cap=" + v4Cap + ") " +
		"{\n (" + v4t2 + ") 116,\n (" + v4t2 + ") 101,\n (" + v4t2 +
		") 115,\n (" + v4t2 + ") 116,\n (" + v4t2 + ") 52,\n (" + v4t2 +
		") 0\n}"
	addDumpTest(v4, "("+v4t+") "+v4s+"\n")

	// C uint8_t array.
	v5, v5l, v5c := testdata.GetCgoUint8tArray()
	v5Len := fmt.Sprintf("%d", v5l)
	v5Cap := fmt.Sprintf("%d", v5c)
	v5t := "[6]testdata._Ctype_uint8_t"
	v5s := "(len=" + v5Len + " cap=" + v5Cap + ") " +
		"{\n 00000000  74 65 73 74 35 00                               " +
		"  |test5.|\n}"
	addDumpTest(v5, "("+v5t+") "+v5s+"\n")

	// C typedefed unsigned char array.
	v6, v6l, v6c := testdata.GetCgoTypdefedUnsignedCharArray()
	v6Len := fmt.Sprintf("%d", v6l)
	v6Cap := fmt.Sprintf("%d", v6c)
	v6t := "[6]testdata._Ctype_custom_uchar_t"
	v6s := "(len=" + v6Len + " cap=" + v6Cap + ") " +
		"{\n 00000000  74 65 73 74 36 00                               " +
		"  |test6.|\n}"
	addDumpTest(v6, "("+v6t+") "+v6s+"\n")
}
