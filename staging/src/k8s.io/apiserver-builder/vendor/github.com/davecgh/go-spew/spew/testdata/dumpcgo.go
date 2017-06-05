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
// command line.  This code should really only be in the dumpcgo_test.go file,
// but unfortunately Go will not allow cgo in test files, so this is a
// workaround to allow cgo types to be tested.  This configuration is used
// because spew itself does not require cgo to run even though it does handle
// certain cgo types specially.  Rather than forcing all clients to require cgo
// and an external C compiler just to run the tests, this scheme makes them
// optional.
// +build cgo,testcgo

package testdata

/*
#include <stdint.h>
typedef unsigned char custom_uchar_t;

char            *ncp = 0;
char            *cp = "test";
char             ca[6] = {'t', 'e', 's', 't', '2', '\0'};
unsigned char    uca[6] = {'t', 'e', 's', 't', '3', '\0'};
signed char      sca[6] = {'t', 'e', 's', 't', '4', '\0'};
uint8_t          ui8ta[6] = {'t', 'e', 's', 't', '5', '\0'};
custom_uchar_t   tuca[6] = {'t', 'e', 's', 't', '6', '\0'};
*/
import "C"

// GetCgoNullCharPointer returns a null char pointer via cgo.  This is only
// used for tests.
func GetCgoNullCharPointer() interface{} {
	return C.ncp
}

// GetCgoCharPointer returns a char pointer via cgo.  This is only used for
// tests.
func GetCgoCharPointer() interface{} {
	return C.cp
}

// GetCgoCharArray returns a char array via cgo and the array's len and cap.
// This is only used for tests.
func GetCgoCharArray() (interface{}, int, int) {
	return C.ca, len(C.ca), cap(C.ca)
}

// GetCgoUnsignedCharArray returns an unsigned char array via cgo and the
// array's len and cap.  This is only used for tests.
func GetCgoUnsignedCharArray() (interface{}, int, int) {
	return C.uca, len(C.uca), cap(C.uca)
}

// GetCgoSignedCharArray returns a signed char array via cgo and the array's len
// and cap.  This is only used for tests.
func GetCgoSignedCharArray() (interface{}, int, int) {
	return C.sca, len(C.sca), cap(C.sca)
}

// GetCgoUint8tArray returns a uint8_t array via cgo and the array's len and
// cap.  This is only used for tests.
func GetCgoUint8tArray() (interface{}, int, int) {
	return C.ui8ta, len(C.ui8ta), cap(C.ui8ta)
}

// GetCgoTypdefedUnsignedCharArray returns a typedefed unsigned char array via
// cgo and the array's len and cap.  This is only used for tests.
func GetCgoTypdefedUnsignedCharArray() (interface{}, int, int) {
	return C.tuca, len(C.tuca), cap(C.tuca)
}
