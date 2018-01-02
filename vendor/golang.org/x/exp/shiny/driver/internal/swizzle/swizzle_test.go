// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package swizzle

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestBGRAShortInput(t *testing.T) {
	const s = "012.456.89A.CDE.GHI.KLM.O"
	testCases := []string{
		0: "012.456.89A.CDE.GHI.KLM.O",
		1: "210.456.89A.CDE.GHI.KLM.O",
		2: "210.654.89A.CDE.GHI.KLM.O",
		3: "210.654.A98.CDE.GHI.KLM.O",
		4: "210.654.A98.EDC.GHI.KLM.O",
		5: "210.654.A98.EDC.IHG.KLM.O",
		6: "210.654.A98.EDC.IHG.MLK.O",
	}
	for i, want := range testCases {
		b := []byte(s)
		BGRA(b[:4*i])
		got := string(b)
		if got != want {
			t.Errorf("i=%d: got %q, want %q", i, got, want)
		}
		changed := got != s
		wantChanged := i != 0
		if changed != wantChanged {
			t.Errorf("i=%d: changed=%t, want %t", i, changed, wantChanged)
		}
	}
}

func TestBGRARandomInput(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	fastBuf := make([]byte, 1024)
	slowBuf := make([]byte, 1024)
	for i := range fastBuf {
		fastBuf[i] = uint8(r.Intn(256))
	}
	copy(slowBuf, fastBuf)

	for i := 0; i < 100000; i++ {
		o := r.Intn(len(fastBuf))
		n := r.Intn(len(fastBuf)-o) &^ 0x03
		BGRA(fastBuf[o : o+n])
		pureGoBGRA(slowBuf[o : o+n])
		if bytes.Equal(fastBuf, slowBuf) {
			continue
		}
		for j := range fastBuf {
			x := fastBuf[j]
			y := slowBuf[j]
			if x != y {
				t.Fatalf("iter %d: swizzling [%d:%d+%d]: bytes differ at offset %d (aka %d+%d): %#02x vs %#02x",
					i, o, o, n, j, o, j-o, x, y)
			}
		}
	}
}

func pureGoBGRA(p []byte) {
	if len(p)%4 != 0 {
		return
	}
	for i := 0; i < len(p); i += 4 {
		p[i+0], p[i+2] = p[i+2], p[i+0]
	}
}

func benchmarkBGRA(b *testing.B, f func([]byte)) {
	const w, h = 1920, 1080 // 1080p RGBA.
	buf := make([]byte, 4*w*h)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f(buf)
	}
}

func BenchmarkBGRA(b *testing.B)       { benchmarkBGRA(b, BGRA) }
func BenchmarkPureGoBGRA(b *testing.B) { benchmarkBGRA(b, pureGoBGRA) }
