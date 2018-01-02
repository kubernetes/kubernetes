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
	"encoding/hex"
	"testing"
)

func TestAesKeyWrap(t *testing.T) {
	// Test vectors from: http://csrc.nist.gov/groups/ST/toolkit/documents/kms/key-wrap.pdf
	kek0, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F")
	cek0, _ := hex.DecodeString("00112233445566778899AABBCCDDEEFF")

	expected0, _ := hex.DecodeString("1FA68B0A8112B447AEF34BD8FB5A7B829D3E862371D2CFE5")

	kek1, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F1011121314151617")
	cek1, _ := hex.DecodeString("00112233445566778899AABBCCDDEEFF")

	expected1, _ := hex.DecodeString("96778B25AE6CA435F92B5B97C050AED2468AB8A17AD84E5D")

	kek2, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F")
	cek2, _ := hex.DecodeString("00112233445566778899AABBCCDDEEFF0001020304050607")

	expected2, _ := hex.DecodeString("A8F9BC1612C68B3FF6E6F4FBE30E71E4769C8B80A32CB8958CD5D17D6B254DA1")

	block0, _ := aes.NewCipher(kek0)
	block1, _ := aes.NewCipher(kek1)
	block2, _ := aes.NewCipher(kek2)

	out0, _ := KeyWrap(block0, cek0)
	out1, _ := KeyWrap(block1, cek1)
	out2, _ := KeyWrap(block2, cek2)

	if bytes.Compare(out0, expected0) != 0 {
		t.Error("output 0 not as expected, got", out0, "wanted", expected0)
	}

	if bytes.Compare(out1, expected1) != 0 {
		t.Error("output 1 not as expected, got", out1, "wanted", expected1)
	}

	if bytes.Compare(out2, expected2) != 0 {
		t.Error("output 2 not as expected, got", out2, "wanted", expected2)
	}

	unwrap0, _ := KeyUnwrap(block0, out0)
	unwrap1, _ := KeyUnwrap(block1, out1)
	unwrap2, _ := KeyUnwrap(block2, out2)

	if bytes.Compare(unwrap0, cek0) != 0 {
		t.Error("key unwrap did not return original input, got", unwrap0, "wanted", cek0)
	}

	if bytes.Compare(unwrap1, cek1) != 0 {
		t.Error("key unwrap did not return original input, got", unwrap1, "wanted", cek1)
	}

	if bytes.Compare(unwrap2, cek2) != 0 {
		t.Error("key unwrap did not return original input, got", unwrap2, "wanted", cek2)
	}
}

func TestAesKeyWrapInvalid(t *testing.T) {
	kek, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F")

	// Invalid unwrap input (bit flipped)
	input0, _ := hex.DecodeString("1EA68C1A8112B447AEF34BD8FB5A7B828D3E862371D2CFE5")

	block, _ := aes.NewCipher(kek)

	_, err := KeyUnwrap(block, input0)
	if err == nil {
		t.Error("key unwrap failed to detect invalid input")
	}

	// Invalid unwrap input (truncated)
	input1, _ := hex.DecodeString("1EA68C1A8112B447AEF34BD8FB5A7B828D3E862371D2CF")

	_, err = KeyUnwrap(block, input1)
	if err == nil {
		t.Error("key unwrap failed to detect truncated input")
	}

	// Invalid wrap input (not multiple of 8)
	input2, _ := hex.DecodeString("0123456789ABCD")

	_, err = KeyWrap(block, input2)
	if err == nil {
		t.Error("key wrap accepted invalid input")
	}

}

func BenchmarkAesKeyWrap(b *testing.B) {
	kek, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F")
	key, _ := hex.DecodeString("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")

	block, _ := aes.NewCipher(kek)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KeyWrap(block, key)
	}
}

func BenchmarkAesKeyUnwrap(b *testing.B) {
	kek, _ := hex.DecodeString("000102030405060708090A0B0C0D0E0F")
	input, _ := hex.DecodeString("1FA68B0A8112B447AEF34BD8FB5A7B829D3E862371D2CFE5")

	block, _ := aes.NewCipher(kek)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KeyUnwrap(block, input)
	}
}
