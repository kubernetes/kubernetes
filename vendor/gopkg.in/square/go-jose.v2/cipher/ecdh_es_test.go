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
	"crypto/ecdsa"
	"crypto/elliptic"
	"encoding/base64"
	"math/big"
	"testing"
)

// Example keys from JWA, Appendix C
var aliceKey = &ecdsa.PrivateKey{
	PublicKey: ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     fromBase64Int("gI0GAILBdu7T53akrFmMyGcsF3n5dO7MmwNBHKW5SV0="),
		Y:     fromBase64Int("SLW_xSffzlPWrHEVI30DHM_4egVwt3NQqeUD7nMFpps="),
	},
	D: fromBase64Int("0_NxaRPUMQoAJt50Gz8YiTr8gRTwyEaCumd-MToTmIo="),
}

var bobKey = &ecdsa.PrivateKey{
	PublicKey: ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     fromBase64Int("weNJy2HscCSM6AEDTDg04biOvhFhyyWvOHQfeF_PxMQ="),
		Y:     fromBase64Int("e8lnCO-AlStT-NJVX-crhB7QRYhiix03illJOVAOyck="),
	},
	D: fromBase64Int("VEmDZpDXXK8p8N0Cndsxs924q6nS1RXFASRl6BfUqdw="),
}

// Build big int from base64-encoded string. Strips whitespace (for testing).
func fromBase64Int(data string) *big.Int {
	val, err := base64.URLEncoding.DecodeString(data)
	if err != nil {
		panic("Invalid test data: " + err.Error())
	}
	return new(big.Int).SetBytes(val)
}

func TestVectorECDHES(t *testing.T) {
	apuData := []byte("Alice")
	apvData := []byte("Bob")

	expected := []byte{
		86, 170, 141, 234, 248, 35, 109, 32, 92, 34, 40, 205, 113, 167, 16, 26}

	output := DeriveECDHES("A128GCM", apuData, apvData, bobKey, &aliceKey.PublicKey, 16)

	if bytes.Compare(output, expected) != 0 {
		t.Error("output did not match what we expect, got", output, "wanted", expected)
	}
}

func TestInvalidECPublicKey(t *testing.T) {
	defer func() { recover() }()

	// Invalid key
	invalid := &ecdsa.PrivateKey{
		PublicKey: ecdsa.PublicKey{
			Curve: elliptic.P256(),
			X:     fromBase64Int("MTEx"),
			Y:     fromBase64Int("MTEx"),
		},
		D: fromBase64Int("0_NxaRPUMQoAJt50Gz8YiTr8gRTwyEaCumd-MToTmIo="),
	}

	DeriveECDHES("A128GCM", []byte{}, []byte{}, bobKey, &invalid.PublicKey, 16)
	t.Fatal("should panic if public key was invalid")
}

func BenchmarkECDHES_128(b *testing.B) {
	apuData := []byte("APU")
	apvData := []byte("APV")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeriveECDHES("ID", apuData, apvData, bobKey, &aliceKey.PublicKey, 16)
	}
}

func BenchmarkECDHES_192(b *testing.B) {
	apuData := []byte("APU")
	apvData := []byte("APV")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeriveECDHES("ID", apuData, apvData, bobKey, &aliceKey.PublicKey, 24)
	}
}

func BenchmarkECDHES_256(b *testing.B) {
	apuData := []byte("APU")
	apvData := []byte("APV")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeriveECDHES("ID", apuData, apvData, bobKey, &aliceKey.PublicKey, 32)
	}
}
