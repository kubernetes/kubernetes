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

package jose

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"errors"
	"io"
	"math/big"
	"testing"
)

func TestVectorsRSA(t *testing.T) {
	// Sources:
	//   http://www.emc.com/emc-plus/rsa-labs/standards-initiatives/pkcs-rsa-cryptography-standard.htm
	//   ftp://ftp.rsa.com/pub/rsalabs/tmp/pkcs1v15crypt-vectors.txt
	priv := &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			N: fromHexInt(`
				a8b3b284af8eb50b387034a860f146c4919f318763cd6c5598c8
				ae4811a1e0abc4c7e0b082d693a5e7fced675cf4668512772c0c
				bc64a742c6c630f533c8cc72f62ae833c40bf25842e984bb78bd
				bf97c0107d55bdb662f5c4e0fab9845cb5148ef7392dd3aaff93
				ae1e6b667bb3d4247616d4f5ba10d4cfd226de88d39f16fb`),
			E: 65537,
		},
		D: fromHexInt(`
				53339cfdb79fc8466a655c7316aca85c55fd8f6dd898fdaf1195
				17ef4f52e8fd8e258df93fee180fa0e4ab29693cd83b152a553d
				4ac4d1812b8b9fa5af0e7f55fe7304df41570926f3311f15c4d6
				5a732c483116ee3d3d2d0af3549ad9bf7cbfb78ad884f84d5beb
				04724dc7369b31def37d0cf539e9cfcdd3de653729ead5d1`),
		Primes: []*big.Int{
			fromHexInt(`
				d32737e7267ffe1341b2d5c0d150a81b586fb3132bed2f8d5262
				864a9cb9f30af38be448598d413a172efb802c21acf1c11c520c
				2f26a471dcad212eac7ca39d`),
			fromHexInt(`
				cc8853d1d54da630fac004f471f281c7b8982d8224a490edbeb3
				3d3e3d5cc93c4765703d1dd791642f1f116a0dd852be2419b2af
				72bfe9a030e860b0288b5d77`),
		},
	}

	input := fromHexBytes(
		"6628194e12073db03ba94cda9ef9532397d50dba79b987004afefe34")

	expectedPKCS := fromHexBytes(`
		50b4c14136bd198c2f3c3ed243fce036e168d56517984a263cd66492b808
		04f169d210f2b9bdfb48b12f9ea05009c77da257cc600ccefe3a6283789d
		8ea0e607ac58e2690ec4ebc10146e8cbaa5ed4d5cce6fe7b0ff9efc1eabb
		564dbf498285f449ee61dd7b42ee5b5892cb90601f30cda07bf26489310b
		cd23b528ceab3c31`)

	expectedOAEP := fromHexBytes(`
		354fe67b4a126d5d35fe36c777791a3f7ba13def484e2d3908aff722fad4
		68fb21696de95d0be911c2d3174f8afcc201035f7b6d8e69402de5451618
		c21a535fa9d7bfc5b8dd9fc243f8cf927db31322d6e881eaa91a996170e6
		57a05a266426d98c88003f8477c1227094a0d9fa1e8c4024309ce1ecccb5
		210035d47ac72e8a`)

	// Mock random reader
	randReader = bytes.NewReader(fromHexBytes(`
		017341ae3875d5f87101f8cc4fa9b9bc156bb04628fccdb2f4f11e905bd3
		a155d376f593bd7304210874eba08a5e22bcccb4c9d3882a93a54db022f5
		03d16338b6b7ce16dc7f4bbf9a96b59772d6606e9747c7649bf9e083db98
		1884a954ab3c6f18b776ea21069d69776a33e96bad48e1dda0a5ef`))
	defer resetRandReader()

	// RSA-PKCS1v1.5 encrypt
	enc := new(rsaEncrypterVerifier)
	enc.publicKey = &priv.PublicKey
	encryptedPKCS, err := enc.encrypt(input, RSA1_5)
	if err != nil {
		t.Error("Encryption failed:", err)
		return
	}

	if bytes.Compare(encryptedPKCS, expectedPKCS) != 0 {
		t.Error("Output does not match expected value (PKCS1v1.5)")
	}

	// RSA-OAEP encrypt
	encryptedOAEP, err := enc.encrypt(input, RSA_OAEP)
	if err != nil {
		t.Error("Encryption failed:", err)
		return
	}

	if bytes.Compare(encryptedOAEP, expectedOAEP) != 0 {
		t.Error("Output does not match expected value (OAEP)")
	}

	// Need fake cipher for PKCS1v1.5 decrypt
	resetRandReader()
	aes := newAESGCM(len(input))

	keygen := randomKeyGenerator{
		size: aes.keySize(),
	}

	// RSA-PKCS1v1.5 decrypt
	dec := new(rsaDecrypterSigner)
	dec.privateKey = priv
	decryptedPKCS, err := dec.decrypt(encryptedPKCS, RSA1_5, keygen)
	if err != nil {
		t.Error("Decryption failed:", err)
		return
	}

	if bytes.Compare(input, decryptedPKCS) != 0 {
		t.Error("Output does not match expected value (PKCS1v1.5)")
	}

	// RSA-OAEP decrypt
	decryptedOAEP, err := dec.decrypt(encryptedOAEP, RSA_OAEP, keygen)
	if err != nil {
		t.Error("decryption failed:", err)
		return
	}

	if bytes.Compare(input, decryptedOAEP) != 0 {
		t.Error("output does not match expected value (OAEP)")
	}
}

func TestEd25519(t *testing.T) {
	_, err := newEd25519Signer("XYZ", nil)
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	enc := new(edEncrypterVerifier)
	enc.publicKey = ed25519PublicKey
	err = enc.verifyPayload([]byte{}, []byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	dec := new(edDecrypterSigner)
	dec.privateKey = ed25519PrivateKey
	_, err = dec.signPayload([]byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	sig, err := dec.signPayload([]byte("This is a test"), "EdDSA")
	if err != nil {
		t.Error("should not error trying to sign payload")
	}
	if sig.Signature == nil {
		t.Error("Check the signature")
	}
	err = enc.verifyPayload([]byte("This is a test"), sig.Signature, "EdDSA")
	if err != nil {
		t.Error("should not error trying to verify payload")
	}

	err = enc.verifyPayload([]byte("This is test number 2"), sig.Signature, "EdDSA")
	if err == nil {
		t.Error("should not error trying to verify payload")
	}
}

func TestInvalidAlgorithmsRSA(t *testing.T) {
	_, err := newRSARecipient("XYZ", nil)
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	_, err = newRSASigner("XYZ", nil)
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	enc := new(rsaEncrypterVerifier)
	enc.publicKey = &rsaTestKey.PublicKey
	_, err = enc.encryptKey([]byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	err = enc.verifyPayload([]byte{}, []byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	dec := new(rsaDecrypterSigner)
	dec.privateKey = rsaTestKey
	_, err = dec.decrypt(make([]byte, 256), "XYZ", randomKeyGenerator{size: 16})
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	_, err = dec.signPayload([]byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}
}

type failingKeyGenerator struct{}

func (ctx failingKeyGenerator) keySize() int {
	return 0
}

func (ctx failingKeyGenerator) genKey() ([]byte, rawHeader, error) {
	return nil, rawHeader{}, errors.New("failed to generate key")
}

func TestPKCSKeyGeneratorFailure(t *testing.T) {
	dec := new(rsaDecrypterSigner)
	dec.privateKey = rsaTestKey
	generator := failingKeyGenerator{}
	_, err := dec.decrypt(make([]byte, 256), RSA1_5, generator)
	if err != ErrCryptoFailure {
		t.Error("should return error on invalid algorithm")
	}
}

func TestInvalidAlgorithmsEC(t *testing.T) {
	_, err := newECDHRecipient("XYZ", nil)
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	_, err = newECDSASigner("XYZ", nil)
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}

	enc := new(ecEncrypterVerifier)
	enc.publicKey = &ecTestKey256.PublicKey
	_, err = enc.encryptKey([]byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Error("should return error on invalid algorithm")
	}
}

func TestInvalidECKeyGen(t *testing.T) {
	gen := ecKeyGenerator{
		size:      16,
		algID:     "A128GCM",
		publicKey: &ecTestKey256.PublicKey,
	}

	if gen.keySize() != 16 {
		t.Error("ec key generator reported incorrect key size")
	}

	_, _, err := gen.genKey()
	if err != nil {
		t.Error("ec key generator failed to generate key", err)
	}
}

func TestInvalidECDecrypt(t *testing.T) {
	dec := ecDecrypterSigner{
		privateKey: ecTestKey256,
	}

	generator := randomKeyGenerator{size: 16}

	// Missing epk header
	headers := rawHeader{}
	headers.set(headerAlgorithm, ECDH_ES)

	_, err := dec.decryptKey(headers, nil, generator)
	if err == nil {
		t.Error("ec decrypter accepted object with missing epk header")
	}

	// Invalid epk header
	headers.set(headerEPK, &JSONWebKey{})

	_, err = dec.decryptKey(headers, nil, generator)
	if err == nil {
		t.Error("ec decrypter accepted object with invalid epk header")
	}
}

func TestDecryptWithIncorrectSize(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Error(err)
		return
	}

	dec := new(rsaDecrypterSigner)
	dec.privateKey = priv
	aes := newAESGCM(16)

	keygen := randomKeyGenerator{
		size: aes.keySize(),
	}

	payload := make([]byte, 254)
	_, err = dec.decrypt(payload, RSA1_5, keygen)
	if err == nil {
		t.Error("Invalid payload size should return error")
	}

	payload = make([]byte, 257)
	_, err = dec.decrypt(payload, RSA1_5, keygen)
	if err == nil {
		t.Error("Invalid payload size should return error")
	}
}

func TestPKCSDecryptNeverFails(t *testing.T) {
	// We don't want RSA-PKCS1 v1.5 decryption to ever fail, in order to prevent
	// side-channel timing attacks (Bleichenbacher attack in particular).
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Error(err)
		return
	}

	dec := new(rsaDecrypterSigner)
	dec.privateKey = priv
	aes := newAESGCM(16)

	keygen := randomKeyGenerator{
		size: aes.keySize(),
	}

	for i := 1; i < 50; i++ {
		payload := make([]byte, 256)
		_, err := io.ReadFull(rand.Reader, payload)
		if err != nil {
			t.Error("Unable to get random data:", err)
			return
		}
		_, err = dec.decrypt(payload, RSA1_5, keygen)
		if err != nil {
			t.Error("PKCS1v1.5 decrypt should never fail:", err)
			return
		}
	}
}

func BenchmarkPKCSDecryptWithValidPayloads(b *testing.B) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}

	enc := new(rsaEncrypterVerifier)
	enc.publicKey = &priv.PublicKey
	dec := new(rsaDecrypterSigner)
	dec.privateKey = priv
	aes := newAESGCM(32)

	b.StopTimer()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plaintext := make([]byte, 32)
		_, err = io.ReadFull(rand.Reader, plaintext)
		if err != nil {
			panic(err)
		}

		ciphertext, err := enc.encrypt(plaintext, RSA1_5)
		if err != nil {
			panic(err)
		}

		keygen := randomKeyGenerator{
			size: aes.keySize(),
		}

		b.StartTimer()
		_, err = dec.decrypt(ciphertext, RSA1_5, keygen)
		b.StopTimer()
		if err != nil {
			panic(err)
		}
	}
}

func BenchmarkPKCSDecryptWithInvalidPayloads(b *testing.B) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}

	enc := new(rsaEncrypterVerifier)
	enc.publicKey = &priv.PublicKey
	dec := new(rsaDecrypterSigner)
	dec.privateKey = priv
	aes := newAESGCM(16)

	keygen := randomKeyGenerator{
		size: aes.keySize(),
	}

	b.StopTimer()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plaintext := make([]byte, 16)
		_, err = io.ReadFull(rand.Reader, plaintext)
		if err != nil {
			panic(err)
		}

		ciphertext, err := enc.encrypt(plaintext, RSA1_5)
		if err != nil {
			panic(err)
		}

		// Do some simple scrambling
		ciphertext[128] ^= 0xFF

		b.StartTimer()
		_, err = dec.decrypt(ciphertext, RSA1_5, keygen)
		b.StopTimer()
		if err != nil {
			panic(err)
		}
	}
}

func TestInvalidEllipticCurve(t *testing.T) {
	signer256 := ecDecrypterSigner{privateKey: ecTestKey256}
	signer384 := ecDecrypterSigner{privateKey: ecTestKey384}
	signer521 := ecDecrypterSigner{privateKey: ecTestKey521}

	_, err := signer256.signPayload([]byte{}, ES384)
	if err == nil {
		t.Error("should not generate ES384 signature with P-256 key")
	}
	_, err = signer256.signPayload([]byte{}, ES512)
	if err == nil {
		t.Error("should not generate ES512 signature with P-256 key")
	}
	_, err = signer384.signPayload([]byte{}, ES256)
	if err == nil {
		t.Error("should not generate ES256 signature with P-384 key")
	}
	_, err = signer384.signPayload([]byte{}, ES512)
	if err == nil {
		t.Error("should not generate ES512 signature with P-384 key")
	}
	_, err = signer521.signPayload([]byte{}, ES256)
	if err == nil {
		t.Error("should not generate ES256 signature with P-521 key")
	}
	_, err = signer521.signPayload([]byte{}, ES384)
	if err == nil {
		t.Error("should not generate ES384 signature with P-521 key")
	}
}

func estInvalidECPublicKey(t *testing.T) {
	// Invalid key
	invalid := &ecdsa.PrivateKey{
		PublicKey: ecdsa.PublicKey{
			Curve: elliptic.P256(),
			X:     fromBase64Int("MTEx"),
			Y:     fromBase64Int("MTEx"),
		},
		D: fromBase64Int("0_NxaRPUMQoAJt50Gz8YiTr8gRTwyEaCumd-MToTmIo"),
	}

	headers := rawHeader{}
	headers.set(headerAlgorithm, ECDH_ES)
	headers.set(headerEPK, &JSONWebKey{
		Key: &invalid.PublicKey,
	})

	dec := ecDecrypterSigner{
		privateKey: ecTestKey256,
	}

	_, err := dec.decryptKey(headers, nil, randomKeyGenerator{size: 16})
	if err == nil {
		t.Fatal("decrypter accepted JWS with invalid ECDH public key")
	}
}

func TestInvalidAlgorithmEC(t *testing.T) {
	err := ecEncrypterVerifier{publicKey: &ecTestKey256.PublicKey}.verifyPayload([]byte{}, []byte{}, "XYZ")
	if err != ErrUnsupportedAlgorithm {
		t.Fatal("should not accept invalid/unsupported algorithm")
	}
}
