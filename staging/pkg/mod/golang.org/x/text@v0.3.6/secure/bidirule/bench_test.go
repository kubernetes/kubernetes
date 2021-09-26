// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bidirule

import (
	"testing"

	"golang.org/x/text/internal/testtext"
)

var benchData = []struct{ name, data string }{
	{"ascii", "Scheveningen"},
	{"arabic", "دبي"},
	{"hangul", "다음과"},
}

func doBench(b *testing.B, fn func(b *testing.B, data string)) {
	for _, d := range benchData {
		testtext.Bench(b, d.name, func(b *testing.B) { fn(b, d.data) })
	}
}

func BenchmarkSpan(b *testing.B) {
	r := New()
	doBench(b, func(b *testing.B, str string) {
		b.SetBytes(int64(len(str)))
		data := []byte(str)
		for i := 0; i < b.N; i++ {
			r.Reset()
			r.Span(data, true)
		}
	})
}

func BenchmarkDirectionASCII(b *testing.B) {
	doBench(b, func(b *testing.B, str string) {
		b.SetBytes(int64(len(str)))
		data := []byte(str)
		for i := 0; i < b.N; i++ {
			Direction(data)
		}
	})
}

func BenchmarkDirectionStringASCII(b *testing.B) {
	doBench(b, func(b *testing.B, str string) {
		b.SetBytes(int64(len(str)))
		for i := 0; i < b.N; i++ {
			DirectionString(str)
		}
	})
}
