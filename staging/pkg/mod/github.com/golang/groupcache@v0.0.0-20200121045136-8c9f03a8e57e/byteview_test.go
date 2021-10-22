/*
Copyright 2012 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package groupcache

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
)

func TestByteView(t *testing.T) {
	for _, s := range []string{"", "x", "yy"} {
		for _, v := range []ByteView{of([]byte(s)), of(s)} {
			name := fmt.Sprintf("string %q, view %+v", s, v)
			if v.Len() != len(s) {
				t.Errorf("%s: Len = %d; want %d", name, v.Len(), len(s))
			}
			if v.String() != s {
				t.Errorf("%s: String = %q; want %q", name, v.String(), s)
			}
			var longDest [3]byte
			if n := v.Copy(longDest[:]); n != len(s) {
				t.Errorf("%s: long Copy = %d; want %d", name, n, len(s))
			}
			var shortDest [1]byte
			if n := v.Copy(shortDest[:]); n != min(len(s), 1) {
				t.Errorf("%s: short Copy = %d; want %d", name, n, min(len(s), 1))
			}
			if got, err := ioutil.ReadAll(v.Reader()); err != nil || string(got) != s {
				t.Errorf("%s: Reader = %q, %v; want %q", name, got, err, s)
			}
			if got, err := ioutil.ReadAll(io.NewSectionReader(v, 0, int64(len(s)))); err != nil || string(got) != s {
				t.Errorf("%s: SectionReader of ReaderAt = %q, %v; want %q", name, got, err, s)
			}
			var dest bytes.Buffer
			if _, err := v.WriteTo(&dest); err != nil || !bytes.Equal(dest.Bytes(), []byte(s)) {
				t.Errorf("%s: WriteTo = %q, %v; want %q", name, dest.Bytes(), err, s)
			}
		}
	}
}

// of returns a byte view of the []byte or string in x.
func of(x interface{}) ByteView {
	if bytes, ok := x.([]byte); ok {
		return ByteView{b: bytes}
	}
	return ByteView{s: x.(string)}
}

func TestByteViewEqual(t *testing.T) {
	tests := []struct {
		a    interface{} // string or []byte
		b    interface{} // string or []byte
		want bool
	}{
		{"x", "x", true},
		{"x", "y", false},
		{"x", "yy", false},
		{[]byte("x"), []byte("x"), true},
		{[]byte("x"), []byte("y"), false},
		{[]byte("x"), []byte("yy"), false},
		{[]byte("x"), "x", true},
		{[]byte("x"), "y", false},
		{[]byte("x"), "yy", false},
		{"x", []byte("x"), true},
		{"x", []byte("y"), false},
		{"x", []byte("yy"), false},
	}
	for i, tt := range tests {
		va := of(tt.a)
		if bytes, ok := tt.b.([]byte); ok {
			if got := va.EqualBytes(bytes); got != tt.want {
				t.Errorf("%d. EqualBytes = %v; want %v", i, got, tt.want)
			}
		} else {
			if got := va.EqualString(tt.b.(string)); got != tt.want {
				t.Errorf("%d. EqualString = %v; want %v", i, got, tt.want)
			}
		}
		if got := va.Equal(of(tt.b)); got != tt.want {
			t.Errorf("%d. Equal = %v; want %v", i, got, tt.want)
		}
	}
}

func TestByteViewSlice(t *testing.T) {
	tests := []struct {
		in   string
		from int
		to   interface{} // nil to mean the end (SliceFrom); else int
		want string
	}{
		{
			in:   "abc",
			from: 1,
			to:   2,
			want: "b",
		},
		{
			in:   "abc",
			from: 1,
			want: "bc",
		},
		{
			in:   "abc",
			to:   2,
			want: "ab",
		},
	}
	for i, tt := range tests {
		for _, v := range []ByteView{of([]byte(tt.in)), of(tt.in)} {
			name := fmt.Sprintf("test %d, view %+v", i, v)
			if tt.to != nil {
				v = v.Slice(tt.from, tt.to.(int))
			} else {
				v = v.SliceFrom(tt.from)
			}
			if v.String() != tt.want {
				t.Errorf("%s: got %q; want %q", name, v.String(), tt.want)
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
