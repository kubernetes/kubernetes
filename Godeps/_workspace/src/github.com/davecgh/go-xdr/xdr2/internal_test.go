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
This test file is part of the xdr package rather than than the xdr_test package
so it can bridge access to the internals to properly test cases which are either
not possible or can't reliably be tested via the public interface. The functions
are only exported while the tests are being run.
*/

package xdr

import (
	"io"
	"reflect"
)

// TstEncode creates a new Encoder to the passed writer and returns the internal
// encode function on the Encoder.
func TstEncode(w io.Writer) func(v reflect.Value) (int, error) {
	enc := NewEncoder(w)
	return enc.encode
}

// TstDecode creates a new Decoder for the passed reader and returns the
// internal decode function on the Decoder.
func TstDecode(r io.Reader) func(v reflect.Value) (int, error) {
	dec := NewDecoder(r)
	return dec.decode
}
