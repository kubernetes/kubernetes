// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rc2

import (
	"testing"
)

func BenchmarkEncrypt(b *testing.B) {
	r, _ := New([]byte{0, 0, 0, 0, 0, 0, 0, 0}, 64)
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Encrypt(src[:], src[:])
	}
}

func BenchmarkDecrypt(b *testing.B) {
	r, _ := New([]byte{0, 0, 0, 0, 0, 0, 0, 0}, 64)
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Decrypt(src[:], src[:])
	}
}
