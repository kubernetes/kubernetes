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

// dummyFmtState implements a fake fmt.State to use for testing invalid
// reflect.Value handling.  This is necessary because the fmt package catches
// invalid values before invoking the formatter on them.
type dummyFmtState struct {
	bytes.Buffer
}

func (dfs *dummyFmtState) Flag(f int) bool {
	return f == int('+')
}

func (dfs *dummyFmtState) Precision() (int, bool) {
	return 0, false
}

func (dfs *dummyFmtState) Width() (int, bool) {
	return 0, false
}

// TestInvalidReflectValue ensures the dump and formatter code handles an
// invalid reflect value properly.  This needs access to internal state since it
// should never happen in real code and therefore can't be tested via the public
// API.
func TestInvalidReflectValue(t *testing.T) {
	i := 1

	// Dump invalid reflect value.
	v := new(reflect.Value)
	buf := new(bytes.Buffer)
	d := dumpState{w: buf, cs: &Config}
	d.dump(*v)
	s := buf.String()
	want := "<invalid>"
	if s != want {
		t.Errorf("InvalidReflectValue #%d\n got: %s want: %s", i, s, want)
	}
	i++

	// Formatter invalid reflect value.
	buf2 := new(dummyFmtState)
	f := formatState{value: *v, cs: &Config, fs: buf2}
	f.format(*v)
	s = buf2.String()
	want = "<invalid>"
	if s != want {
		t.Errorf("InvalidReflectValue #%d got: %s want: %s", i, s, want)
	}
}

// SortValues makes the internal sortValues function available to the test
// package.
func SortValues(values []reflect.Value, cs *ConfigState) {
	sortValues(values, cs)
}
