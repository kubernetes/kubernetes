// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package salsa20

import (
	"bytes"
	"encoding/hex"
	"testing"
)

func fromHex(s string) []byte {
	ret, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return ret
}

// testVectors was taken from set 6 of the ECRYPT test vectors:
// http://www.ecrypt.eu.org/stream/svn/viewcvs.cgi/ecrypt/trunk/submissions/salsa20/full/verified.test-vectors?logsort=rev&rev=210&view=markup
var testVectors = []struct {
	key      []byte
	iv       []byte
	numBytes int
	xor      []byte
}{
	{
		fromHex("0053A6F94C9FF24598EB3E91E4378ADD3083D6297CCF2275C81B6EC11467BA0D"),
		fromHex("0D74DB42A91077DE"),
		131072,
		fromHex("C349B6A51A3EC9B712EAED3F90D8BCEE69B7628645F251A996F55260C62EF31FD6C6B0AEA94E136C9D984AD2DF3578F78E457527B03A0450580DD874F63B1AB9"),
	},
	{
		fromHex("0558ABFE51A4F74A9DF04396E93C8FE23588DB2E81D4277ACD2073C6196CBF12"),
		fromHex("167DE44BB21980E7"),
		131072,
		fromHex("C3EAAF32836BACE32D04E1124231EF47E101367D6305413A0EEB07C60698A2876E4D031870A739D6FFDDD208597AFF0A47AC17EDB0167DD67EBA84F1883D4DFD"),
	},
	{
		fromHex("0A5DB00356A9FC4FA2F5489BEE4194E73A8DE03386D92C7FD22578CB1E71C417"),
		fromHex("1F86ED54BB2289F0"),
		131072,
		fromHex("3CD23C3DC90201ACC0CF49B440B6C417F0DC8D8410A716D5314C059E14B1A8D9A9FB8EA3D9C8DAE12B21402F674AA95C67B1FC514E994C9D3F3A6E41DFF5BBA6"),
	},
	{
		fromHex("0F62B5085BAE0154A7FA4DA0F34699EC3F92E5388BDE3184D72A7DD02376C91C"),
		fromHex("288FF65DC42B92F9"),
		131072,
		fromHex("E00EBCCD70D69152725F9987982178A2E2E139C7BCBE04CA8A0E99E318D9AB76F988C8549F75ADD790BA4F81C176DA653C1A043F11A958E169B6D2319F4EEC1A"),
	},
}

func TestSalsa20(t *testing.T) {
	var inBuf, outBuf []byte
	var key [32]byte

	for i, test := range testVectors {
		if test.numBytes%64 != 0 {
			t.Errorf("#%d: numBytes is not a multiple of 64", i)
			continue
		}

		if test.numBytes > len(inBuf) {
			inBuf = make([]byte, test.numBytes)
			outBuf = make([]byte, test.numBytes)
		}
		in := inBuf[:test.numBytes]
		out := outBuf[:test.numBytes]
		copy(key[:], test.key)
		XORKeyStream(out, in, test.iv, &key)

		var xor [64]byte
		for len(out) > 0 {
			for i := 0; i < 64; i++ {
				xor[i] ^= out[i]
			}
			out = out[64:]
		}

		if !bytes.Equal(xor[:], test.xor) {
			t.Errorf("#%d: bad result", i)
		}
	}
}

var xSalsa20TestData = []struct {
	in, nonce, key, out []byte
}{
	{
		[]byte("Hello world!"),
		[]byte("24-byte nonce for xsalsa"),
		[]byte("this is 32-byte key for xsalsa20"),
		[]byte{0x00, 0x2d, 0x45, 0x13, 0x84, 0x3f, 0xc2, 0x40, 0xc4, 0x01, 0xe5, 0x41},
	},
	{
		make([]byte, 64),
		[]byte("24-byte nonce for xsalsa"),
		[]byte("this is 32-byte key for xsalsa20"),
		[]byte{0x48, 0x48, 0x29, 0x7f, 0xeb, 0x1f, 0xb5, 0x2f, 0xb6,
			0x6d, 0x81, 0x60, 0x9b, 0xd5, 0x47, 0xfa, 0xbc, 0xbe, 0x70,
			0x26, 0xed, 0xc8, 0xb5, 0xe5, 0xe4, 0x49, 0xd0, 0x88, 0xbf,
			0xa6, 0x9c, 0x08, 0x8f, 0x5d, 0x8d, 0xa1, 0xd7, 0x91, 0x26,
			0x7c, 0x2c, 0x19, 0x5a, 0x7f, 0x8c, 0xae, 0x9c, 0x4b, 0x40,
			0x50, 0xd0, 0x8c, 0xe6, 0xd3, 0xa1, 0x51, 0xec, 0x26, 0x5f,
			0x3a, 0x58, 0xe4, 0x76, 0x48},
	},
}

func TestXSalsa20(t *testing.T) {
	var key [32]byte

	for i, test := range xSalsa20TestData {
		out := make([]byte, len(test.in))
		copy(key[:], test.key)
		XORKeyStream(out, test.in, test.nonce, &key)
		if !bytes.Equal(out, test.out) {
			t.Errorf("%d: expected %x, got %x", i, test.out, out)
		}
	}
}

var (
	keyArray [32]byte
	key      = &keyArray
	nonce    [8]byte
	msg      = make([]byte, 1<<10)
)

func BenchmarkXOR1K(b *testing.B) {
	b.StopTimer()
	out := make([]byte, 1024)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		XORKeyStream(out, msg[:1024], nonce[:], key)
	}
	b.SetBytes(1024)
}
