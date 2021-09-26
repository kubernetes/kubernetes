// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.7
// +build go1.7

package precis

import (
	"testing"

	"golang.org/x/text/internal/testtext"
)

var benchData = []struct{ name, str string }{
	{"ASCII", "Malvolio"},
	{"NotNormalized", "abcdefg\u0301\u031f"},
	{"Arabic", "دبي"},
	{"Hangul", "동일조건변경허락"},
}

var benchProfiles = []struct {
	name string
	p    *Profile
}{
	{"FreeForm", NewFreeform()},
	{"Nickname", Nickname},
	{"OpaqueString", OpaqueString},
	{"UsernameCaseMapped", UsernameCaseMapped},
	{"UsernameCasePreserved", UsernameCasePreserved},
}

func doBench(b *testing.B, f func(b *testing.B, p *Profile, s string)) {
	for _, bp := range benchProfiles {
		for _, d := range benchData {
			testtext.Bench(b, bp.name+"/"+d.name, func(b *testing.B) {
				f(b, bp.p, d.str)
			})
		}
	}
}

func BenchmarkString(b *testing.B) {
	doBench(b, func(b *testing.B, p *Profile, s string) {
		for i := 0; i < b.N; i++ {
			p.String(s)
		}
	})
}

func BenchmarkBytes(b *testing.B) {
	doBench(b, func(b *testing.B, p *Profile, s string) {
		src := []byte(s)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.Bytes(src)
		}
	})
}

func BenchmarkAppend(b *testing.B) {
	doBench(b, func(b *testing.B, p *Profile, s string) {
		src := []byte(s)
		dst := make([]byte, 0, 4096)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.Append(dst, src)
		}
	})
}

func BenchmarkTransform(b *testing.B) {
	doBench(b, func(b *testing.B, p *Profile, s string) {
		src := []byte(s)
		dst := make([]byte, 2*len(s))
		t := p.NewTransformer()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			t.Transform(dst, src, true)
		}
	})
}
