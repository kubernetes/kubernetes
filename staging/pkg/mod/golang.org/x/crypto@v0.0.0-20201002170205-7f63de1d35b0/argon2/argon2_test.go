// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package argon2

import (
	"bytes"
	"encoding/hex"
	"testing"
)

var (
	genKatPassword = []byte{
		0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
		0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
		0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
		0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
	}
	genKatSalt   = []byte{0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02}
	genKatSecret = []byte{0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03}
	genKatAAD    = []byte{0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}
)

func TestArgon2(t *testing.T) {
	defer func(sse4 bool) { useSSE4 = sse4 }(useSSE4)

	if useSSE4 {
		t.Log("SSE4.1 version")
		testArgon2i(t)
		testArgon2d(t)
		testArgon2id(t)
		useSSE4 = false
	}
	t.Log("generic version")
	testArgon2i(t)
	testArgon2d(t)
	testArgon2id(t)
}

func testArgon2d(t *testing.T) {
	want := []byte{
		0x51, 0x2b, 0x39, 0x1b, 0x6f, 0x11, 0x62, 0x97,
		0x53, 0x71, 0xd3, 0x09, 0x19, 0x73, 0x42, 0x94,
		0xf8, 0x68, 0xe3, 0xbe, 0x39, 0x84, 0xf3, 0xc1,
		0xa1, 0x3a, 0x4d, 0xb9, 0xfa, 0xbe, 0x4a, 0xcb,
	}
	hash := deriveKey(argon2d, genKatPassword, genKatSalt, genKatSecret, genKatAAD, 3, 32, 4, 32)
	if !bytes.Equal(hash, want) {
		t.Errorf("derived key does not match - got: %s , want: %s", hex.EncodeToString(hash), hex.EncodeToString(want))
	}
}

func testArgon2i(t *testing.T) {
	want := []byte{
		0xc8, 0x14, 0xd9, 0xd1, 0xdc, 0x7f, 0x37, 0xaa,
		0x13, 0xf0, 0xd7, 0x7f, 0x24, 0x94, 0xbd, 0xa1,
		0xc8, 0xde, 0x6b, 0x01, 0x6d, 0xd3, 0x88, 0xd2,
		0x99, 0x52, 0xa4, 0xc4, 0x67, 0x2b, 0x6c, 0xe8,
	}
	hash := deriveKey(argon2i, genKatPassword, genKatSalt, genKatSecret, genKatAAD, 3, 32, 4, 32)
	if !bytes.Equal(hash, want) {
		t.Errorf("derived key does not match - got: %s , want: %s", hex.EncodeToString(hash), hex.EncodeToString(want))
	}
}

func testArgon2id(t *testing.T) {
	want := []byte{
		0x0d, 0x64, 0x0d, 0xf5, 0x8d, 0x78, 0x76, 0x6c,
		0x08, 0xc0, 0x37, 0xa3, 0x4a, 0x8b, 0x53, 0xc9,
		0xd0, 0x1e, 0xf0, 0x45, 0x2d, 0x75, 0xb6, 0x5e,
		0xb5, 0x25, 0x20, 0xe9, 0x6b, 0x01, 0xe6, 0x59,
	}
	hash := deriveKey(argon2id, genKatPassword, genKatSalt, genKatSecret, genKatAAD, 3, 32, 4, 32)
	if !bytes.Equal(hash, want) {
		t.Errorf("derived key does not match - got: %s , want: %s", hex.EncodeToString(hash), hex.EncodeToString(want))
	}
}

func TestVectors(t *testing.T) {
	password, salt := []byte("password"), []byte("somesalt")
	for i, v := range testVectors {
		want, err := hex.DecodeString(v.hash)
		if err != nil {
			t.Fatalf("Test %d: failed to decode hash: %v", i, err)
		}
		hash := deriveKey(v.mode, password, salt, nil, nil, v.time, v.memory, v.threads, uint32(len(want)))
		if !bytes.Equal(hash, want) {
			t.Errorf("Test %d - got: %s want: %s", i, hex.EncodeToString(hash), hex.EncodeToString(want))
		}
	}
}

func benchmarkArgon2(mode int, time, memory uint32, threads uint8, keyLen uint32, b *testing.B) {
	password := []byte("password")
	salt := []byte("choosing random salts is hard")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		deriveKey(mode, password, salt, nil, nil, time, memory, threads, keyLen)
	}
}

func BenchmarkArgon2i(b *testing.B) {
	b.Run(" Time: 3 Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2i, 3, 32*1024, 1, 32, b) })
	b.Run(" Time: 4 Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2i, 4, 32*1024, 1, 32, b) })
	b.Run(" Time: 5 Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2i, 5, 32*1024, 1, 32, b) })
	b.Run(" Time: 3 Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2i, 3, 64*1024, 4, 32, b) })
	b.Run(" Time: 4 Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2i, 4, 64*1024, 4, 32, b) })
	b.Run(" Time: 5 Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2i, 5, 64*1024, 4, 32, b) })
}

func BenchmarkArgon2d(b *testing.B) {
	b.Run(" Time: 3, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2d, 3, 32*1024, 1, 32, b) })
	b.Run(" Time: 4, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2d, 4, 32*1024, 1, 32, b) })
	b.Run(" Time: 5, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2d, 5, 32*1024, 1, 32, b) })
	b.Run(" Time: 3, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2d, 3, 64*1024, 4, 32, b) })
	b.Run(" Time: 4, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2d, 4, 64*1024, 4, 32, b) })
	b.Run(" Time: 5, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2d, 5, 64*1024, 4, 32, b) })
}

func BenchmarkArgon2id(b *testing.B) {
	b.Run(" Time: 3, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2id, 3, 32*1024, 1, 32, b) })
	b.Run(" Time: 4, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2id, 4, 32*1024, 1, 32, b) })
	b.Run(" Time: 5, Memory: 32 MB, Threads: 1", func(b *testing.B) { benchmarkArgon2(argon2id, 5, 32*1024, 1, 32, b) })
	b.Run(" Time: 3, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2id, 3, 64*1024, 4, 32, b) })
	b.Run(" Time: 4, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2id, 4, 64*1024, 4, 32, b) })
	b.Run(" Time: 5, Memory: 64 MB, Threads: 4", func(b *testing.B) { benchmarkArgon2(argon2id, 5, 64*1024, 4, 32, b) })
}

// Generated with the CLI of https://github.com/P-H-C/phc-winner-argon2/blob/master/argon2-specs.pdf
var testVectors = []struct {
	mode         int
	time, memory uint32
	threads      uint8
	hash         string
}{
	{
		mode: argon2i, time: 1, memory: 64, threads: 1,
		hash: "b9c401d1844a67d50eae3967dc28870b22e508092e861a37",
	},
	{
		mode: argon2d, time: 1, memory: 64, threads: 1,
		hash: "8727405fd07c32c78d64f547f24150d3f2e703a89f981a19",
	},
	{
		mode: argon2id, time: 1, memory: 64, threads: 1,
		hash: "655ad15eac652dc59f7170a7332bf49b8469be1fdb9c28bb",
	},
	{
		mode: argon2i, time: 2, memory: 64, threads: 1,
		hash: "8cf3d8f76a6617afe35fac48eb0b7433a9a670ca4a07ed64",
	},
	{
		mode: argon2d, time: 2, memory: 64, threads: 1,
		hash: "3be9ec79a69b75d3752acb59a1fbb8b295a46529c48fbb75",
	},
	{
		mode: argon2id, time: 2, memory: 64, threads: 1,
		hash: "068d62b26455936aa6ebe60060b0a65870dbfa3ddf8d41f7",
	},
	{
		mode: argon2i, time: 2, memory: 64, threads: 2,
		hash: "2089f3e78a799720f80af806553128f29b132cafe40d059f",
	},
	{
		mode: argon2d, time: 2, memory: 64, threads: 2,
		hash: "68e2462c98b8bc6bb60ec68db418ae2c9ed24fc6748a40e9",
	},
	{
		mode: argon2id, time: 2, memory: 64, threads: 2,
		hash: "350ac37222f436ccb5c0972f1ebd3bf6b958bf2071841362",
	},
	{
		mode: argon2i, time: 3, memory: 256, threads: 2,
		hash: "f5bbf5d4c3836af13193053155b73ec7476a6a2eb93fd5e6",
	},
	{
		mode: argon2d, time: 3, memory: 256, threads: 2,
		hash: "f4f0669218eaf3641f39cc97efb915721102f4b128211ef2",
	},
	{
		mode: argon2id, time: 3, memory: 256, threads: 2,
		hash: "4668d30ac4187e6878eedeacf0fd83c5a0a30db2cc16ef0b",
	},
	{
		mode: argon2i, time: 4, memory: 4096, threads: 4,
		hash: "a11f7b7f3f93f02ad4bddb59ab62d121e278369288a0d0e7",
	},
	{
		mode: argon2d, time: 4, memory: 4096, threads: 4,
		hash: "935598181aa8dc2b720914aa6435ac8d3e3a4210c5b0fb2d",
	},
	{
		mode: argon2id, time: 4, memory: 4096, threads: 4,
		hash: "145db9733a9f4ee43edf33c509be96b934d505a4efb33c5a",
	},
	{
		mode: argon2i, time: 4, memory: 1024, threads: 8,
		hash: "0cdd3956aa35e6b475a7b0c63488822f774f15b43f6e6e17",
	},
	{
		mode: argon2d, time: 4, memory: 1024, threads: 8,
		hash: "83604fc2ad0589b9d055578f4d3cc55bc616df3578a896e9",
	},
	{
		mode: argon2id, time: 4, memory: 1024, threads: 8,
		hash: "8dafa8e004f8ea96bf7c0f93eecf67a6047476143d15577f",
	},
	{
		mode: argon2i, time: 2, memory: 64, threads: 3,
		hash: "5cab452fe6b8479c8661def8cd703b611a3905a6d5477fe6",
	},
	{
		mode: argon2d, time: 2, memory: 64, threads: 3,
		hash: "22474a423bda2ccd36ec9afd5119e5c8949798cadf659f51",
	},
	{
		mode: argon2id, time: 2, memory: 64, threads: 3,
		hash: "4a15b31aec7c2590b87d1f520be7d96f56658172deaa3079",
	},
	{
		mode: argon2i, time: 3, memory: 1024, threads: 6,
		hash: "d236b29c2b2a09babee842b0dec6aa1e83ccbdea8023dced",
	},
	{
		mode: argon2d, time: 3, memory: 1024, threads: 6,
		hash: "a3351b0319a53229152023d9206902f4ef59661cdca89481",
	},
	{
		mode: argon2id, time: 3, memory: 1024, threads: 6,
		hash: "1640b932f4b60e272f5d2207b9a9c626ffa1bd88d2349016",
	},
}
