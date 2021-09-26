// Copyright (c) 2013-2016 Dave Collins <dave@davec.name>

// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.

// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

// NOTE: Due to the following build constraints, this file will only be compiled
// when the code is not running on Google App Engine, compiled by GopherJS, and
// "-tags safe" is not added to the go build command line.  The "disableunsafe"
// tag is deprecated and thus should not be used.
// +build !js,!appengine,!safe,!disableunsafe,go1.4

/*
This test file is part of the spew package rather than than the spew_test
package because it needs access to internals to properly test certain cases
which are not possible via the public interface since they should never happen.
*/

package spew

import (
	"bytes"
	"reflect"
	"testing"
)

// changeKind uses unsafe to intentionally change the kind of a reflect.Value to
// the maximum kind value which does not exist.  This is needed to test the
// fallback code which punts to the standard fmt library for new types that
// might get added to the language.
func changeKind(v *reflect.Value, readOnly bool) {
	flags := flagField(v)
	if readOnly {
		*flags |= flagRO
	} else {
		*flags &^= flagRO
	}
	*flags |= flagKindMask
}

// TestAddedReflectValue tests functionaly of the dump and formatter code which
// falls back to the standard fmt library for new types that might get added to
// the language.
func TestAddedReflectValue(t *testing.T) {
	i := 1

	// Dump using a reflect.Value that is exported.
	v := reflect.ValueOf(int8(5))
	changeKind(&v, false)
	buf := new(bytes.Buffer)
	d := dumpState{w: buf, cs: &Config}
	d.dump(v)
	s := buf.String()
	want := "(int8) 5"
	if s != want {
		t.Errorf("TestAddedReflectValue #%d\n got: %s want: %s", i, s, want)
	}
	i++

	// Dump using a reflect.Value that is not exported.
	changeKind(&v, true)
	buf.Reset()
	d.dump(v)
	s = buf.String()
	want = "(int8) <int8 Value>"
	if s != want {
		t.Errorf("TestAddedReflectValue #%d\n got: %s want: %s", i, s, want)
	}
	i++

	// Formatter using a reflect.Value that is exported.
	changeKind(&v, false)
	buf2 := new(dummyFmtState)
	f := formatState{value: v, cs: &Config, fs: buf2}
	f.format(v)
	s = buf2.String()
	want = "5"
	if s != want {
		t.Errorf("TestAddedReflectValue #%d got: %s want: %s", i, s, want)
	}
	i++

	// Formatter using a reflect.Value that is not exported.
	changeKind(&v, true)
	buf2.Reset()
	f = formatState{value: v, cs: &Config, fs: buf2}
	f.format(v)
	s = buf2.String()
	want = "<int8 Value>"
	if s != want {
		t.Errorf("TestAddedReflectValue #%d got: %s want: %s", i, s, want)
	}
}
