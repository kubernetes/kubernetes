// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || purego || !gc
// +build !amd64 purego !gc

package salsa

// XORKeyStream crypts bytes from in to out using the given key and counters.
// In and out must overlap entirely or not at all. Counter
// contains the raw salsa20 counter bytes (both nonce and block counter).
func XORKeyStream(out, in []byte, counter *[16]byte, key *[32]byte) {
	genericXORKeyStream(out, in, counter, key)
}
