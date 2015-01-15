/*
 * Copyright (c) 2014 Dave Collins <dave@davec.name>
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

package xdr_test

import (
	"testing"

	. "github.com/davecgh/go-xdr/xdr2"
)

// TestErrorCodeStringer tests the stringized output for the ErrorCode type.
func TestErrorCodeStringer(t *testing.T) {
	tests := []struct {
		in   ErrorCode
		want string
	}{
		{ErrBadArguments, "ErrBadArguments"},
		{ErrUnsupportedType, "ErrUnsupportedType"},
		{ErrBadEnumValue, "ErrBadEnumValue"},
		{ErrNotSettable, "ErrNotSettable"},
		{ErrOverflow, "ErrOverflow"},
		{ErrNilInterface, "ErrNilInterface"},
		{ErrIO, "ErrIO"},
		{ErrParseTime, "ErrParseTime"},
		{0xffff, "Unknown ErrorCode (65535)"},
	}

	for i, test := range tests {
		result := test.in.String()
		if result != test.want {
			t.Errorf("String #%d\n got: %s want: %s", i, result,
				test.want)
			continue
		}
	}
}

// TestUnmarshalError tests the error output for the UnmarshalError type.
func TestUnmarshalError(t *testing.T) {
	tests := []struct {
		in   UnmarshalError
		want string
	}{
		{
			UnmarshalError{
				ErrorCode:   ErrIO,
				Func:        "test",
				Description: "EOF while decoding 5 bytes",
				Value:       "testval",
			},
			"xdr:test: EOF while decoding 5 bytes - read: 'testval'",
		},
		{
			UnmarshalError{
				ErrorCode:   ErrBadEnumValue,
				Func:        "test",
				Description: "invalid enum",
				Value:       "testenum",
			},
			"xdr:test: invalid enum - read: 'testenum'",
		},
		{
			UnmarshalError{
				ErrorCode:   ErrNilInterface,
				Func:        "test",
				Description: "can't unmarshal to nil interface",
				Value:       nil,
			},
			"xdr:test: can't unmarshal to nil interface",
		},
	}

	for i, test := range tests {
		result := test.in.Error()
		if result != test.want {
			t.Errorf("Error #%d\n got: %s want: %s", i, result,
				test.want)
			continue
		}
	}
}

// TestMarshalError tests the error output for the MarshalError type.
func TestMarshalError(t *testing.T) {
	tests := []struct {
		in   MarshalError
		want string
	}{
		{
			MarshalError{
				ErrorCode:   ErrIO,
				Func:        "test",
				Description: "EOF while encoding 5 bytes",
				Value:       []byte{0x01, 0x02},
			},
			"xdr:test: EOF while encoding 5 bytes - wrote: '[1 2]'",
		},
		{
			MarshalError{
				ErrorCode:   ErrBadEnumValue,
				Func:        "test",
				Description: "invalid enum",
				Value:       "testenum",
			},
			"xdr:test: invalid enum - value: 'testenum'",
		},
		{
			MarshalError{
				ErrorCode:   ErrNilInterface,
				Func:        "test",
				Description: "can't marshal to nil interface",
				Value:       nil,
			},
			"xdr:test: can't marshal to nil interface",
		},
	}

	for i, test := range tests {
		result := test.in.Error()
		if result != test.want {
			t.Errorf("Error #%d\n got: %s want: %s", i, result,
				test.want)
			continue
		}
	}
}
