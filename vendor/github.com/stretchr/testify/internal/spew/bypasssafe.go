// Copyright (c) 2015-2016 Dave Collins <dave@davec.name>
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
// when the code is running on Google App Engine, compiled by GopherJS, or
// "-tags safe" is added to the go build command line.  The "disableunsafe"
// tag is deprecated and thus should not be used.
//go:build js || appengine || safe || disableunsafe || !go1.4
// +build js appengine safe disableunsafe !go1.4

package spew

import "reflect"

const (
	// UnsafeDisabled is a build-time constant which specifies whether or
	// not access to the unsafe package is available.
	UnsafeDisabled = true
)

// unsafeReflectValue typically converts the passed reflect.Value into a one
// that bypasses the typical safety restrictions preventing access to
// unaddressable and unexported data.  However, doing this relies on access to
// the unsafe package.  This is a stub version which simply returns the passed
// reflect.Value when the unsafe package is not available.
func unsafeReflectValue(v reflect.Value) reflect.Value {
	return v
}
