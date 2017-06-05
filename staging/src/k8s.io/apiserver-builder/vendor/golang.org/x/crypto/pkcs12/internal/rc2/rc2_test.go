// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rc2

import (
	"bytes"
	"encoding/hex"
	"testing"
)

func TestEncryptDecrypt(t *testing.T) {

	// TODO(dgryski): add the rest of the test vectors from the RFC
	var tests = []struct {
		key    string
		plain  string
		cipher string
		t1     int
	}{
		{
			"0000000000000000",
			"0000000000000000",
			"ebb773f993278eff",
			63,
		},
		{
			"ffffffffffffffff",
			"ffffffffffffffff",
			"278b27e42e2f0d49",
			64,
		},
		{
			"3000000000000000",
			"1000000000000001",
			"30649edf9be7d2c2",
			64,
		},
		{
			"88",
			"0000000000000000",
			"61a8a244adacccf0",
			64,
		},
		{
			"88bca90e90875a",
			"0000000000000000",
			"6ccf4308974c267f",
			64,
		},
		{
			"88bca90e90875a7f0f79c384627bafb2",
			"0000000000000000",
			"1a807d272bbe5db1",
			64,
		},
		{
			"88bca90e90875a7f0f79c384627bafb2",
			"0000000000000000",
			"2269552ab0f85ca6",
			128,
		},
		{
			"88bca90e90875a7f0f79c384627bafb216f80a6f85920584c42fceb0be255daf1e",
			"0000000000000000",
			"5b78d3a43dfff1f1",
			129,
		},
	}

	for _, tt := range tests {
		k, _ := hex.DecodeString(tt.key)
		p, _ := hex.DecodeString(tt.plain)
		c, _ := hex.DecodeString(tt.cipher)

		b, _ := New(k, tt.t1)

		var dst [8]byte

		b.Encrypt(dst[:], p)

		if !bytes.Equal(dst[:], c) {
			t.Errorf("encrypt failed: got % 2x wanted % 2x\n", dst, c)
		}

		b.Decrypt(dst[:], c)

		if !bytes.Equal(dst[:], p) {
			t.Errorf("decrypt failed: got % 2x wanted % 2x\n", dst, p)
		}
	}
}
