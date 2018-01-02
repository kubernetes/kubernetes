/*-
 * Copyright 2014 Square Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package josecipher

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"io"
	"strings"
	"testing"
)

func TestInvalidInputs(t *testing.T) {
	key := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	}

	nonce := []byte{
		92, 80, 104, 49, 133, 25, 161, 215, 173, 101, 219, 211, 136, 91, 210, 145}

	aead, _ := NewCBCHMAC(key, aes.NewCipher)
	ciphertext := aead.Seal(nil, nonce, []byte("plaintext"), []byte("aad"))

	// Changed AAD, must fail
	_, err := aead.Open(nil, nonce, ciphertext, []byte("INVALID"))
	if err == nil {
		t.Error("must detect invalid aad")
	}

	// Empty ciphertext, must fail
	_, err = aead.Open(nil, nonce, []byte{}, []byte("aad"))
	if err == nil {
		t.Error("must detect invalid/empty ciphertext")
	}

	// Corrupt ciphertext, must fail
	corrupt := make([]byte, len(ciphertext))
	copy(corrupt, ciphertext)
	corrupt[0] ^= 0xFF

	_, err = aead.Open(nil, nonce, corrupt, []byte("aad"))
	if err == nil {
		t.Error("must detect corrupt ciphertext")
	}

	// Corrupt authtag, must fail
	copy(corrupt, ciphertext)
	corrupt[len(ciphertext)-1] ^= 0xFF

	_, err = aead.Open(nil, nonce, corrupt, []byte("aad"))
	if err == nil {
		t.Error("must detect corrupt authtag")
	}

	// Truncated data, must fail
	_, err = aead.Open(nil, nonce, ciphertext[:10], []byte("aad"))
	if err == nil {
		t.Error("must detect corrupt authtag")
	}
}

func TestVectorsAESCBC128(t *testing.T) {
	// Source: http://tools.ietf.org/html/draft-ietf-jose-json-web-encryption-29#appendix-A.2
	plaintext := []byte{
		76, 105, 118, 101, 32, 108, 111, 110, 103, 32, 97, 110, 100, 32,
		112, 114, 111, 115, 112, 101, 114, 46}

	aad := []byte{
		101, 121, 74, 104, 98, 71, 99, 105, 79, 105, 74, 83, 85, 48, 69,
		120, 88, 122, 85, 105, 76, 67, 74, 108, 98, 109, 77, 105, 79, 105,
		74, 66, 77, 84, 73, 52, 81, 48, 74, 68, 76, 85, 104, 84, 77, 106, 85,
		50, 73, 110, 48}

	expectedCiphertext := []byte{
		40, 57, 83, 181, 119, 33, 133, 148, 198, 185, 243, 24, 152, 230, 6,
		75, 129, 223, 127, 19, 210, 82, 183, 230, 168, 33, 215, 104, 143,
		112, 56, 102}

	expectedAuthtag := []byte{
		246, 17, 244, 190, 4, 95, 98, 3, 231, 0, 115, 157, 242, 203, 100,
		191}

	key := []byte{
		4, 211, 31, 197, 84, 157, 252, 254, 11, 100, 157, 250, 63, 170, 106, 206,
		107, 124, 212, 45, 111, 107, 9, 219, 200, 177, 0, 240, 143, 156, 44, 207}

	nonce := []byte{
		3, 22, 60, 12, 43, 67, 104, 105, 108, 108, 105, 99, 111, 116, 104, 101}

	enc, err := NewCBCHMAC(key, aes.NewCipher)
	out := enc.Seal(nil, nonce, plaintext, aad)
	if err != nil {
		t.Error("Unable to encrypt:", err)
		return
	}

	if bytes.Compare(out[:len(out)-16], expectedCiphertext) != 0 {
		t.Error("Ciphertext did not match")
	}
	if bytes.Compare(out[len(out)-16:], expectedAuthtag) != 0 {
		t.Error("Auth tag did not match")
	}
}

func TestVectorsAESCBC256(t *testing.T) {
	// Source: https://tools.ietf.org/html/draft-mcgrew-aead-aes-cbc-hmac-sha2-05#section-5.4
	plaintext := []byte{
		0x41, 0x20, 0x63, 0x69, 0x70, 0x68, 0x65, 0x72, 0x20, 0x73, 0x79, 0x73, 0x74, 0x65, 0x6d, 0x20,
		0x6d, 0x75, 0x73, 0x74, 0x20, 0x6e, 0x6f, 0x74, 0x20, 0x62, 0x65, 0x20, 0x72, 0x65, 0x71, 0x75,
		0x69, 0x72, 0x65, 0x64, 0x20, 0x74, 0x6f, 0x20, 0x62, 0x65, 0x20, 0x73, 0x65, 0x63, 0x72, 0x65,
		0x74, 0x2c, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x69, 0x74, 0x20, 0x6d, 0x75, 0x73, 0x74, 0x20, 0x62,
		0x65, 0x20, 0x61, 0x62, 0x6c, 0x65, 0x20, 0x74, 0x6f, 0x20, 0x66, 0x61, 0x6c, 0x6c, 0x20, 0x69,
		0x6e, 0x74, 0x6f, 0x20, 0x74, 0x68, 0x65, 0x20, 0x68, 0x61, 0x6e, 0x64, 0x73, 0x20, 0x6f, 0x66,
		0x20, 0x74, 0x68, 0x65, 0x20, 0x65, 0x6e, 0x65, 0x6d, 0x79, 0x20, 0x77, 0x69, 0x74, 0x68, 0x6f,
		0x75, 0x74, 0x20, 0x69, 0x6e, 0x63, 0x6f, 0x6e, 0x76, 0x65, 0x6e, 0x69, 0x65, 0x6e, 0x63, 0x65}

	aad := []byte{
		0x54, 0x68, 0x65, 0x20, 0x73, 0x65, 0x63, 0x6f, 0x6e, 0x64, 0x20, 0x70, 0x72, 0x69, 0x6e, 0x63,
		0x69, 0x70, 0x6c, 0x65, 0x20, 0x6f, 0x66, 0x20, 0x41, 0x75, 0x67, 0x75, 0x73, 0x74, 0x65, 0x20,
		0x4b, 0x65, 0x72, 0x63, 0x6b, 0x68, 0x6f, 0x66, 0x66, 0x73}

	expectedCiphertext := []byte{
		0x4a, 0xff, 0xaa, 0xad, 0xb7, 0x8c, 0x31, 0xc5, 0xda, 0x4b, 0x1b, 0x59, 0x0d, 0x10, 0xff, 0xbd,
		0x3d, 0xd8, 0xd5, 0xd3, 0x02, 0x42, 0x35, 0x26, 0x91, 0x2d, 0xa0, 0x37, 0xec, 0xbc, 0xc7, 0xbd,
		0x82, 0x2c, 0x30, 0x1d, 0xd6, 0x7c, 0x37, 0x3b, 0xcc, 0xb5, 0x84, 0xad, 0x3e, 0x92, 0x79, 0xc2,
		0xe6, 0xd1, 0x2a, 0x13, 0x74, 0xb7, 0x7f, 0x07, 0x75, 0x53, 0xdf, 0x82, 0x94, 0x10, 0x44, 0x6b,
		0x36, 0xeb, 0xd9, 0x70, 0x66, 0x29, 0x6a, 0xe6, 0x42, 0x7e, 0xa7, 0x5c, 0x2e, 0x08, 0x46, 0xa1,
		0x1a, 0x09, 0xcc, 0xf5, 0x37, 0x0d, 0xc8, 0x0b, 0xfe, 0xcb, 0xad, 0x28, 0xc7, 0x3f, 0x09, 0xb3,
		0xa3, 0xb7, 0x5e, 0x66, 0x2a, 0x25, 0x94, 0x41, 0x0a, 0xe4, 0x96, 0xb2, 0xe2, 0xe6, 0x60, 0x9e,
		0x31, 0xe6, 0xe0, 0x2c, 0xc8, 0x37, 0xf0, 0x53, 0xd2, 0x1f, 0x37, 0xff, 0x4f, 0x51, 0x95, 0x0b,
		0xbe, 0x26, 0x38, 0xd0, 0x9d, 0xd7, 0xa4, 0x93, 0x09, 0x30, 0x80, 0x6d, 0x07, 0x03, 0xb1, 0xf6}

	expectedAuthtag := []byte{
		0x4d, 0xd3, 0xb4, 0xc0, 0x88, 0xa7, 0xf4, 0x5c, 0x21, 0x68, 0x39, 0x64, 0x5b, 0x20, 0x12, 0xbf,
		0x2e, 0x62, 0x69, 0xa8, 0xc5, 0x6a, 0x81, 0x6d, 0xbc, 0x1b, 0x26, 0x77, 0x61, 0x95, 0x5b, 0xc5}

	key := []byte{
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f}

	nonce := []byte{
		0x1a, 0xf3, 0x8c, 0x2d, 0xc2, 0xb9, 0x6f, 0xfd, 0xd8, 0x66, 0x94, 0x09, 0x23, 0x41, 0xbc, 0x04}

	enc, err := NewCBCHMAC(key, aes.NewCipher)
	out := enc.Seal(nil, nonce, plaintext, aad)
	if err != nil {
		t.Error("Unable to encrypt:", err)
		return
	}

	if bytes.Compare(out[:len(out)-32], expectedCiphertext) != 0 {
		t.Error("Ciphertext did not match, got", out[:len(out)-32], "wanted", expectedCiphertext)
	}
	if bytes.Compare(out[len(out)-32:], expectedAuthtag) != 0 {
		t.Error("Auth tag did not match, got", out[len(out)-32:], "wanted", expectedAuthtag)
	}
}

func TestAESCBCRoundtrip(t *testing.T) {
	key128 := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

	key192 := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7}

	key256 := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

	nonce := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

	RunRoundtrip(t, key128, nonce)
	RunRoundtrip(t, key192, nonce)
	RunRoundtrip(t, key256, nonce)
}

func RunRoundtrip(t *testing.T, key, nonce []byte) {
	aead, err := NewCBCHMAC(key, aes.NewCipher)
	if err != nil {
		panic(err)
	}

	if aead.NonceSize() != len(nonce) {
		panic("invalid nonce")
	}

	// Test pre-existing data in dst buffer
	dst := []byte{15, 15, 15, 15}
	plaintext := []byte{0, 0, 0, 0}
	aad := []byte{4, 3, 2, 1}

	result := aead.Seal(dst, nonce, plaintext, aad)
	if bytes.Compare(dst, result[:4]) != 0 {
		t.Error("Existing data in dst not preserved")
	}

	// Test pre-existing (empty) dst buffer with sufficient capacity
	dst = make([]byte, 256)[:0]
	result, err = aead.Open(dst, nonce, result[4:], aad)
	if err != nil {
		panic(err)
	}

	if bytes.Compare(result, plaintext) != 0 {
		t.Error("Plaintext does not match output")
	}
}

func TestAESCBCOverhead(t *testing.T) {
	aead, err := NewCBCHMAC(make([]byte, 32), aes.NewCipher)
	if err != nil {
		panic(err)
	}

	if aead.Overhead() != 32 {
		t.Error("CBC-HMAC reports incorrect overhead value")
	}
}

func TestPadding(t *testing.T) {
	for i := 0; i < 256; i++ {
		slice := make([]byte, i)
		padded := padBuffer(slice, 16)
		if len(padded)%16 != 0 {
			t.Error("failed to pad slice properly", i)
			return
		}
		unpadded, err := unpadBuffer(padded, 16)
		if err != nil || len(unpadded) != i {
			t.Error("failed to unpad slice properly", i)
			return
		}
	}
}

func TestInvalidKey(t *testing.T) {
	key := make([]byte, 30)
	_, err := NewCBCHMAC(key, aes.NewCipher)
	if err == nil {
		t.Error("should not be able to instantiate CBC-HMAC with invalid key")
	}
}

func TestTruncatedCiphertext(t *testing.T) {
	key := make([]byte, 32)
	nonce := make([]byte, 16)
	data := make([]byte, 32)

	io.ReadFull(rand.Reader, key)
	io.ReadFull(rand.Reader, nonce)

	aead, err := NewCBCHMAC(key, aes.NewCipher)
	if err != nil {
		panic(err)
	}

	ctx := aead.(*cbcAEAD)
	ct := aead.Seal(nil, nonce, data, nil)

	// Truncated ciphertext, but with correct auth tag
	truncated, tail := resize(ct[:len(ct)-ctx.authtagBytes-2], uint64(len(ct))-2)
	copy(tail, ctx.computeAuthTag(nil, nonce, truncated[:len(truncated)-ctx.authtagBytes]))

	// Open should fail
	_, err = aead.Open(nil, nonce, truncated, nil)
	if err == nil {
		t.Error("open on truncated ciphertext should fail")
	}
}

func TestInvalidPaddingOpen(t *testing.T) {
	key := make([]byte, 32)
	nonce := make([]byte, 16)

	// Plaintext with invalid padding
	plaintext := padBuffer(make([]byte, 28), aes.BlockSize)
	plaintext[len(plaintext)-1] = 0xFF

	io.ReadFull(rand.Reader, key)
	io.ReadFull(rand.Reader, nonce)

	block, _ := aes.NewCipher(key)
	cbc := cipher.NewCBCEncrypter(block, nonce)
	buffer := append([]byte{}, plaintext...)
	cbc.CryptBlocks(buffer, buffer)

	aead, _ := NewCBCHMAC(key, aes.NewCipher)
	ctx := aead.(*cbcAEAD)

	// Mutated ciphertext, but with correct auth tag
	size := uint64(len(buffer))
	ciphertext, tail := resize(buffer, size+(uint64(len(key))/2))
	copy(tail, ctx.computeAuthTag(nil, nonce, ciphertext[:size]))

	// Open should fail (b/c of invalid padding, even though tag matches)
	_, err := aead.Open(nil, nonce, ciphertext, nil)
	if err == nil || !strings.Contains(err.Error(), "invalid padding") {
		t.Error("no or unexpected error on open with invalid padding:", err)
	}
}

func TestInvalidPadding(t *testing.T) {
	for i := 0; i < 256; i++ {
		slice := make([]byte, i)
		padded := padBuffer(slice, 16)
		if len(padded)%16 != 0 {
			t.Error("failed to pad slice properly", i)
			return
		}

		paddingBytes := 16 - (i % 16)

		// Mutate padding for testing
		for j := 1; j <= paddingBytes; j++ {
			mutated := make([]byte, len(padded))
			copy(mutated, padded)
			mutated[len(mutated)-j] ^= 0xFF

			_, err := unpadBuffer(mutated, 16)
			if err == nil {
				t.Error("unpad on invalid padding should fail", i)
				return
			}
		}

		// Test truncated padding
		_, err := unpadBuffer(padded[:len(padded)-1], 16)
		if err == nil {
			t.Error("unpad on truncated padding should fail", i)
			return
		}
	}
}

func TestZeroLengthPadding(t *testing.T) {
	data := make([]byte, 16)
	data, err := unpadBuffer(data, 16)
	if err == nil {
		t.Error("padding with 0x00 should never be valid")
	}
}

func benchEncryptCBCHMAC(b *testing.B, keySize, chunkSize int) {
	key := make([]byte, keySize*2)
	nonce := make([]byte, 16)

	io.ReadFull(rand.Reader, key)
	io.ReadFull(rand.Reader, nonce)

	chunk := make([]byte, chunkSize)

	aead, err := NewCBCHMAC(key, aes.NewCipher)
	if err != nil {
		panic(err)
	}

	b.SetBytes(int64(chunkSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aead.Seal(nil, nonce, chunk, nil)
	}
}

func benchDecryptCBCHMAC(b *testing.B, keySize, chunkSize int) {
	key := make([]byte, keySize*2)
	nonce := make([]byte, 16)

	io.ReadFull(rand.Reader, key)
	io.ReadFull(rand.Reader, nonce)

	chunk := make([]byte, chunkSize)

	aead, err := NewCBCHMAC(key, aes.NewCipher)
	if err != nil {
		panic(err)
	}

	out := aead.Seal(nil, nonce, chunk, nil)

	b.SetBytes(int64(chunkSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aead.Open(nil, nonce, out, nil)
	}
}

func BenchmarkEncryptAES128_CBCHMAC_1k(b *testing.B) {
	benchEncryptCBCHMAC(b, 16, 1024)
}

func BenchmarkEncryptAES128_CBCHMAC_64k(b *testing.B) {
	benchEncryptCBCHMAC(b, 16, 65536)
}

func BenchmarkEncryptAES128_CBCHMAC_1MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 16, 1048576)
}

func BenchmarkEncryptAES128_CBCHMAC_64MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 16, 67108864)
}

func BenchmarkDecryptAES128_CBCHMAC_1k(b *testing.B) {
	benchDecryptCBCHMAC(b, 16, 1024)
}

func BenchmarkDecryptAES128_CBCHMAC_64k(b *testing.B) {
	benchDecryptCBCHMAC(b, 16, 65536)
}

func BenchmarkDecryptAES128_CBCHMAC_1MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 16, 1048576)
}

func BenchmarkDecryptAES128_CBCHMAC_64MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 16, 67108864)
}

func BenchmarkEncryptAES192_CBCHMAC_64k(b *testing.B) {
	benchEncryptCBCHMAC(b, 24, 65536)
}

func BenchmarkEncryptAES192_CBCHMAC_1MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 24, 1048576)
}

func BenchmarkEncryptAES192_CBCHMAC_64MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 24, 67108864)
}

func BenchmarkDecryptAES192_CBCHMAC_1k(b *testing.B) {
	benchDecryptCBCHMAC(b, 24, 1024)
}

func BenchmarkDecryptAES192_CBCHMAC_64k(b *testing.B) {
	benchDecryptCBCHMAC(b, 24, 65536)
}

func BenchmarkDecryptAES192_CBCHMAC_1MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 24, 1048576)
}

func BenchmarkDecryptAES192_CBCHMAC_64MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 24, 67108864)
}

func BenchmarkEncryptAES256_CBCHMAC_64k(b *testing.B) {
	benchEncryptCBCHMAC(b, 32, 65536)
}

func BenchmarkEncryptAES256_CBCHMAC_1MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 32, 1048576)
}

func BenchmarkEncryptAES256_CBCHMAC_64MB(b *testing.B) {
	benchEncryptCBCHMAC(b, 32, 67108864)
}

func BenchmarkDecryptAES256_CBCHMAC_1k(b *testing.B) {
	benchDecryptCBCHMAC(b, 32, 1032)
}

func BenchmarkDecryptAES256_CBCHMAC_64k(b *testing.B) {
	benchDecryptCBCHMAC(b, 32, 65536)
}

func BenchmarkDecryptAES256_CBCHMAC_1MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 32, 1048576)
}

func BenchmarkDecryptAES256_CBCHMAC_64MB(b *testing.B) {
	benchDecryptCBCHMAC(b, 32, 67108864)
}
