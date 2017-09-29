// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!appengine,!gccgo

package salsa

// This function is implemented in salsa2020_amd64.s.

//go:noescape

func salsa2020XORKeyStream(out, in *byte, n uint64, nonce, key *byte)

// XORKeyStream crypts bytes from in to out using the given key and counters.
// In and out may be the same slice but otherwise should not overlap. Counter
// contains the raw salsa20 counter bytes (both nonce and block counter).
func XORKeyStream(out, in []byte, counter *[16]byte, key *[32]byte) {
	if len(in) == 0 {
		return
	}
	_ = out[len(in)-1]
	salsa2020XORKeyStream(&out[0], &in[0], uint64(len(in)), &counter[0], &key[0])
}
