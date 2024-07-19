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
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"encoding/binary"
)

// DeriveECDHES derives a shared encryption key using ECDH/ConcatKDF as described in JWE/JWA.
// It is an error to call this function with a private/public key that are not on the same
// curve. Callers must ensure that the keys are valid before calling this function. Output
// size may be at most 1<<16 bytes (64 KiB).
func DeriveECDHES(alg string, apuData, apvData []byte, priv *ecdsa.PrivateKey, pub *ecdsa.PublicKey, size int) []byte {
	if size > 1<<16 {
		panic("ECDH-ES output size too large, must be less than or equal to 1<<16")
	}

	// algId, partyUInfo, partyVInfo inputs must be prefixed with the length
	algID := lengthPrefixed([]byte(alg))
	ptyUInfo := lengthPrefixed(apuData)
	ptyVInfo := lengthPrefixed(apvData)

	// suppPubInfo is the encoded length of the output size in bits
	supPubInfo := make([]byte, 4)
	binary.BigEndian.PutUint32(supPubInfo, uint32(size)*8)

	if !priv.PublicKey.Curve.IsOnCurve(pub.X, pub.Y) {
		panic("public key not on same curve as private key")
	}

	z, _ := priv.Curve.ScalarMult(pub.X, pub.Y, priv.D.Bytes())
	zBytes := z.Bytes()

	// Note that calling z.Bytes() on a big.Int may strip leading zero bytes from
	// the returned byte array. This can lead to a problem where zBytes will be
	// shorter than expected which breaks the key derivation. Therefore we must pad
	// to the full length of the expected coordinate here before calling the KDF.
	octSize := dSize(priv.Curve)
	if len(zBytes) != octSize {
		zBytes = append(bytes.Repeat([]byte{0}, octSize-len(zBytes)), zBytes...)
	}

	reader := NewConcatKDF(crypto.SHA256, zBytes, algID, ptyUInfo, ptyVInfo, supPubInfo, []byte{})
	key := make([]byte, size)

	// Read on the KDF will never fail
	_, _ = reader.Read(key)

	return key
}

// dSize returns the size in octets for a coordinate on a elliptic curve.
func dSize(curve elliptic.Curve) int {
	order := curve.Params().P
	bitLen := order.BitLen()
	size := bitLen / 8
	if bitLen%8 != 0 {
		size++
	}
	return size
}

func lengthPrefixed(data []byte) []byte {
	out := make([]byte, len(data)+4)
	binary.BigEndian.PutUint32(out, uint32(len(data)))
	copy(out[4:], data)
	return out
}
