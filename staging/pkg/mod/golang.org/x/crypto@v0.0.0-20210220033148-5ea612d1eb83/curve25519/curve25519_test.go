// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package curve25519

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"testing"
)

const expectedHex = "89161fde887b2b53de549af483940106ecc114d6982daa98256de23bdf77661a"

func TestX25519Basepoint(t *testing.T) {
	x := make([]byte, 32)
	x[0] = 1

	for i := 0; i < 200; i++ {
		var err error
		x, err = X25519(x, Basepoint)
		if err != nil {
			t.Fatal(err)
		}
	}

	result := fmt.Sprintf("%x", x)
	if result != expectedHex {
		t.Errorf("incorrect result: got %s, want %s", result, expectedHex)
	}
}

func TestLowOrderPoints(t *testing.T) {
	scalar := make([]byte, ScalarSize)
	if _, err := rand.Read(scalar); err != nil {
		t.Fatal(err)
	}
	for i, p := range lowOrderPoints {
		out, err := X25519(scalar, p)
		if err == nil {
			t.Errorf("%d: expected error, got nil", i)
		}
		if out != nil {
			t.Errorf("%d: expected nil output, got %x", i, out)
		}
	}
}

func TestTestVectors(t *testing.T) {
	t.Run("Generic", func(t *testing.T) { testTestVectors(t, scalarMultGeneric) })
	t.Run("Native", func(t *testing.T) { testTestVectors(t, ScalarMult) })
	t.Run("X25519", func(t *testing.T) {
		testTestVectors(t, func(dst, scalar, point *[32]byte) {
			out, err := X25519(scalar[:], point[:])
			if err != nil {
				t.Fatal(err)
			}
			copy(dst[:], out)
		})
	})
}

func testTestVectors(t *testing.T, scalarMult func(dst, scalar, point *[32]byte)) {
	for _, tv := range testVectors {
		var got [32]byte
		scalarMult(&got, &tv.In, &tv.Base)
		if !bytes.Equal(got[:], tv.Expect[:]) {
			t.Logf("    in = %x", tv.In)
			t.Logf("  base = %x", tv.Base)
			t.Logf("   got = %x", got)
			t.Logf("expect = %x", tv.Expect)
			t.Fail()
		}
	}
}

// TestHighBitIgnored tests the following requirement in RFC 7748:
//
//	When receiving such an array, implementations of X25519 (but not X448) MUST
//	mask the most significant bit in the final byte.
//
// Regression test for issue #30095.
func TestHighBitIgnored(t *testing.T) {
	var s, u [32]byte
	rand.Read(s[:])
	rand.Read(u[:])

	var hi0, hi1 [32]byte

	u[31] &= 0x7f
	ScalarMult(&hi0, &s, &u)

	u[31] |= 0x80
	ScalarMult(&hi1, &s, &u)

	if !bytes.Equal(hi0[:], hi1[:]) {
		t.Errorf("high bit of group point should not affect result")
	}
}

func BenchmarkScalarBaseMult(b *testing.B) {
	var in, out [32]byte
	in[0] = 1

	b.SetBytes(32)
	for i := 0; i < b.N; i++ {
		ScalarBaseMult(&out, &in)
	}
}
