/*
 * Copyright (c) 2012 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package xdr_test

import (
	"github.com/davecgh/go-xdr/xdr"
	"testing"
	"unsafe"
)

func BenchmarkUnmarshal(b *testing.B) {
	b.StopTimer()
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}
	// XDR encoded data described by the above structure.
	encodedData := []byte{
		0xAB, 0xCD, 0xEF, 0x00,
		0x00, 0x00, 0x00, 0x02,
		0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x0A}
	var h ImageHeader
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, _ = xdr.Unmarshal(encodedData, &h)
	}
	b.SetBytes(int64(len(encodedData)))
}

func BenchmarkMarshal(b *testing.B) {
	b.StopTimer()
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}
	h := ImageHeader{[3]byte{0xAB, 0xCD, 0xEF}, 2, true, 10}
	size := unsafe.Sizeof(h)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, _ = xdr.Marshal(&h)
	}
	b.SetBytes(int64(size))
}
