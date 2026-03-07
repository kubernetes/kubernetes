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
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/subtle"
	"errors"
	"fmt"
	"hash"
	"io"

	josecipher "github.com/go-jose/go-jose/v4/cipher"
)

// RandReader is a cryptographically secure random number generator (stubbed out in tests).
var RandReader = rand.Reader

const (
	// RFC7518 recommends a minimum of 1,000 iterations:
	// 	- https://tools.ietf.org/html/rfc7518#section-4.8.1.2
	//
	// NIST recommends a minimum of 10,000:
	// 	- https://pages.nist.gov/800-63-3/sp800-63b.html
	//
	// 1Password increased in 2023 from 100,000 to 650,000:
	//  - https://support.1password.com/pbkdf2/
	//
	// OWASP recommended 600,000 in Dec 2022:
	//	- https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html#pbkdf2
	defaultP2C = 600000
	// Default salt size: 128 bits
	defaultP2SSize = 16
)

// Dummy key cipher for shared symmetric key mode
type symmetricKeyCipher struct {
	key []byte // Pre-shared content-encryption key
	p2c int    // PBES2 Count
	p2s []byte // PBES2 Salt Input
}

// Signer/verifier for MAC modes
type symmetricMac struct {
	key []byte
}

// Input/output from an AEAD operation
type aeadParts struct {
	iv, ciphertext, tag []byte
}

// A content cipher based on an AEAD construction
type aeadContentCipher struct {
	keyBytes     int
	authtagBytes int
	getAead      func(key []byte) (cipher.AEAD, error)
}

// Random key generator
type randomKeyGenerator struct {
	size int
}

// Static key generator
type staticKeyGenerator struct {
	key []byte
}

// Create a new content cipher based on AES-GCM
func newAESGCM(keySize int) contentCipher {
	return &aeadContentCipher{
		keyBytes:     keySize,
		authtagBytes: 16,
		getAead: func(key []byte) (cipher.AEAD, error) {
			aes, err := aes.NewCipher(key)
			if err != nil {
				return nil, err
			}

			return cipher.NewGCM(aes)
		},
	}
}

// Create a new content cipher based on AES-CBC+HMAC
func newAESCBC(keySize int) contentCipher {
	return &aeadContentCipher{
		keyBytes:     keySize * 2,
		authtagBytes: keySize,
		getAead: func(key []byte) (cipher.AEAD, error) {
			return josecipher.NewCBCHMAC(key, aes.NewCipher)
		},
	}
}

// Get an AEAD cipher object for the given content encryption algorithm
func getContentCipher(alg ContentEncryption) contentCipher {
	switch alg {
	case A128GCM:
		return newAESGCM(16)
	case A192GCM:
		return newAESGCM(24)
	case A256GCM:
		return newAESGCM(32)
	case A128CBC_HS256:
		return newAESCBC(16)
	case A192CBC_HS384:
		return newAESCBC(24)
	case A256CBC_HS512:
		return newAESCBC(32)
	default:
		return nil
	}
}

// getPbkdf2Params returns the key length and hash function used in
// pbkdf2.Key.
func getPbkdf2Params(alg KeyAlgorithm) (int, func() hash.Hash) {
	switch alg {
	case PBES2_HS256_A128KW:
		return 16, sha256.New
	case PBES2_HS384_A192KW:
		return 24, sha512.New384
	case PBES2_HS512_A256KW:
		return 32, sha512.New
	default:
		panic("invalid algorithm")
	}
}

// getRandomSalt generates a new salt of the given size.
func getRandomSalt(size int) ([]byte, error) {
	salt := make([]byte, size)
	_, err := io.ReadFull(RandReader, salt)
	if err != nil {
		return nil, err
	}

	return salt, nil
}

// newSymmetricRecipient creates a JWE encrypter based on AES-GCM key wrap.
func newSymmetricRecipient(keyAlg KeyAlgorithm, key []byte) (recipientKeyInfo, error) {
	switch keyAlg {
	case DIRECT, A128GCMKW, A192GCMKW, A256GCMKW, A128KW, A192KW, A256KW:
	case PBES2_HS256_A128KW, PBES2_HS384_A192KW, PBES2_HS512_A256KW:
	default:
		return recipientKeyInfo{}, ErrUnsupportedAlgorithm
	}

	return recipientKeyInfo{
		keyAlg: keyAlg,
		keyEncrypter: &symmetricKeyCipher{
			key: key,
		},
	}, nil
}

// newSymmetricSigner creates a recipientSigInfo based on the given key.
func newSymmetricSigner(sigAlg SignatureAlgorithm, key []byte) (recipientSigInfo, error) {
	// Verify that key management algorithm is supported by this encrypter
	switch sigAlg {
	case HS256, HS384, HS512:
	default:
		return recipientSigInfo{}, ErrUnsupportedAlgorithm
	}

	return recipientSigInfo{
		sigAlg: sigAlg,
		signer: &symmetricMac{
			key: key,
		},
	}, nil
}

// Generate a random key for the given content cipher
func (ctx randomKeyGenerator) genKey() ([]byte, rawHeader, error) {
	key := make([]byte, ctx.size)
	_, err := io.ReadFull(RandReader, key)
	if err != nil {
		return nil, rawHeader{}, err
	}

	return key, rawHeader{}, nil
}

// Key size for random generator
func (ctx randomKeyGenerator) keySize() int {
	return ctx.size
}

// Generate a static key (for direct mode)
func (ctx staticKeyGenerator) genKey() ([]byte, rawHeader, error) {
	cek := make([]byte, len(ctx.key))
	copy(cek, ctx.key)
	return cek, rawHeader{}, nil
}

// Key size for static generator
func (ctx staticKeyGenerator) keySize() int {
	return len(ctx.key)
}

// Get key size for this cipher
func (ctx aeadContentCipher) keySize() int {
	return ctx.keyBytes
}

// Encrypt some data
func (ctx aeadContentCipher) encrypt(key, aad, pt []byte) (*aeadParts, error) {
	// Get a new AEAD instance
	aead, err := ctx.getAead(key)
	if err != nil {
		return nil, err
	}

	// Initialize a new nonce
	iv := make([]byte, aead.NonceSize())
	_, err = io.ReadFull(RandReader, iv)
	if err != nil {
		return nil, err
	}

	ciphertextAndTag := aead.Seal(nil, iv, pt, aad)
	offset := len(ciphertextAndTag) - ctx.authtagBytes

	return &aeadParts{
		iv:         iv,
		ciphertext: ciphertextAndTag[:offset],
		tag:        ciphertextAndTag[offset:],
	}, nil
}

// Decrypt some data
func (ctx aeadContentCipher) decrypt(key, aad []byte, parts *aeadParts) ([]byte, error) {
	aead, err := ctx.getAead(key)
	if err != nil {
		return nil, err
	}

	if len(parts.iv) != aead.NonceSize() || len(parts.tag) < ctx.authtagBytes {
		return nil, ErrCryptoFailure
	}

	return aead.Open(nil, parts.iv, append(parts.ciphertext, parts.tag...), aad)
}

// Encrypt the content encryption key.
func (ctx *symmetricKeyCipher) encryptKey(cek []byte, alg KeyAlgorithm) (recipientInfo, error) {
	switch alg {
	case DIRECT:
		return recipientInfo{
			header: &rawHeader{},
		}, nil
	case A128GCMKW, A192GCMKW, A256GCMKW:
		aead := newAESGCM(len(ctx.key))

		parts, err := aead.encrypt(ctx.key, []byte{}, cek)
		if err != nil {
			return recipientInfo{}, err
		}

		header := &rawHeader{}

		if err = header.set(headerIV, newBuffer(parts.iv)); err != nil {
			return recipientInfo{}, err
		}

		if err = header.set(headerTag, newBuffer(parts.tag)); err != nil {
			return recipientInfo{}, err
		}

		return recipientInfo{
			header:       header,
			encryptedKey: parts.ciphertext,
		}, nil
	case A128KW, A192KW, A256KW:
		block, err := aes.NewCipher(ctx.key)
		if err != nil {
			return recipientInfo{}, err
		}

		jek, err := josecipher.KeyWrap(block, cek)
		if err != nil {
			return recipientInfo{}, err
		}

		return recipientInfo{
			encryptedKey: jek,
			header:       &rawHeader{},
		}, nil
	case PBES2_HS256_A128KW, PBES2_HS384_A192KW, PBES2_HS512_A256KW:
		if len(ctx.p2s) == 0 {
			salt, err := getRandomSalt(defaultP2SSize)
			if err != nil {
				return recipientInfo{}, err
			}
			ctx.p2s = salt
		}

		if ctx.p2c <= 0 {
			ctx.p2c = defaultP2C
		}

		// salt is UTF8(Alg) || 0x00 || Salt Input
		salt := bytes.Join([][]byte{[]byte(alg), ctx.p2s}, []byte{0x00})

		// derive key
		keyLen, h := getPbkdf2Params(alg)
		key, err := pbkdf2.Key(h, string(ctx.key), salt, ctx.p2c, keyLen)
		if err != nil {
			return recipientInfo{}, nil
		}

		// use AES cipher with derived key
		block, err := aes.NewCipher(key)
		if err != nil {
			return recipientInfo{}, err
		}

		jek, err := josecipher.KeyWrap(block, cek)
		if err != nil {
			return recipientInfo{}, err
		}

		header := &rawHeader{}

		if err = header.set(headerP2C, ctx.p2c); err != nil {
			return recipientInfo{}, err
		}

		if err = header.set(headerP2S, newBuffer(ctx.p2s)); err != nil {
			return recipientInfo{}, err
		}

		return recipientInfo{
			encryptedKey: jek,
			header:       header,
		}, nil
	}

	return recipientInfo{}, ErrUnsupportedAlgorithm
}

// Decrypt the content encryption key.
func (ctx *symmetricKeyCipher) decryptKey(headers rawHeader, recipient *recipientInfo, generator keyGenerator) ([]byte, error) {
	switch headers.getAlgorithm() {
	case DIRECT:
		cek := make([]byte, len(ctx.key))
		copy(cek, ctx.key)
		return cek, nil
	case A128GCMKW, A192GCMKW, A256GCMKW:
		aead := newAESGCM(len(ctx.key))

		iv, err := headers.getIV()
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid IV: %v", err)
		}
		tag, err := headers.getTag()
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid tag: %v", err)
		}

		parts := &aeadParts{
			iv:         iv.bytes(),
			ciphertext: recipient.encryptedKey,
			tag:        tag.bytes(),
		}

		cek, err := aead.decrypt(ctx.key, []byte{}, parts)
		if err != nil {
			return nil, err
		}

		return cek, nil
	case A128KW, A192KW, A256KW:
		block, err := aes.NewCipher(ctx.key)
		if err != nil {
			return nil, err
		}

		cek, err := josecipher.KeyUnwrap(block, recipient.encryptedKey)
		if err != nil {
			return nil, err
		}
		return cek, nil
	case PBES2_HS256_A128KW, PBES2_HS384_A192KW, PBES2_HS512_A256KW:
		p2s, err := headers.getP2S()
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid P2S: %v", err)
		}
		if p2s == nil || len(p2s.data) == 0 {
			return nil, fmt.Errorf("go-jose/go-jose: invalid P2S: must be present")
		}

		p2c, err := headers.getP2C()
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid P2C: %v", err)
		}
		if p2c <= 0 {
			return nil, fmt.Errorf("go-jose/go-jose: invalid P2C: must be a positive integer")
		}
		if p2c > 1000000 {
			// An unauthenticated attacker can set a high P2C value. Set an upper limit to avoid
			// DoS attacks.
			return nil, fmt.Errorf("go-jose/go-jose: invalid P2C: too high")
		}

		// salt is UTF8(Alg) || 0x00 || Salt Input
		alg := headers.getAlgorithm()
		salt := bytes.Join([][]byte{[]byte(alg), p2s.bytes()}, []byte{0x00})

		// derive key
		keyLen, h := getPbkdf2Params(alg)
		key, err := pbkdf2.Key(h, string(ctx.key), salt, p2c, keyLen)
		if err != nil {
			return nil, err
		}

		// use AES cipher with derived key
		block, err := aes.NewCipher(key)
		if err != nil {
			return nil, err
		}

		cek, err := josecipher.KeyUnwrap(block, recipient.encryptedKey)
		if err != nil {
			return nil, err
		}
		return cek, nil
	}

	return nil, ErrUnsupportedAlgorithm
}

// Sign the given payload
func (ctx symmetricMac) signPayload(payload []byte, alg SignatureAlgorithm) (Signature, error) {
	mac, err := ctx.hmac(payload, alg)
	if err != nil {
		return Signature{}, err
	}

	return Signature{
		Signature: mac,
		protected: &rawHeader{},
	}, nil
}

// Verify the given payload
func (ctx symmetricMac) verifyPayload(payload []byte, mac []byte, alg SignatureAlgorithm) error {
	expected, err := ctx.hmac(payload, alg)
	if err != nil {
		return errors.New("go-jose/go-jose: failed to compute hmac")
	}

	if len(mac) != len(expected) {
		return errors.New("go-jose/go-jose: invalid hmac")
	}

	match := subtle.ConstantTimeCompare(mac, expected)
	if match != 1 {
		return errors.New("go-jose/go-jose: invalid hmac")
	}

	return nil
}

// Compute the HMAC based on the given alg value
func (ctx symmetricMac) hmac(payload []byte, alg SignatureAlgorithm) ([]byte, error) {
	var hash func() hash.Hash

	// https://datatracker.ietf.org/doc/html/rfc7518#section-3.2
	// A key of the same size as the hash output (for instance, 256 bits for
	// "HS256") or larger MUST be used
	switch alg {
	case HS256:
		if len(ctx.key)*8 < 256 {
			return nil, ErrInvalidKeySize
		}
		hash = sha256.New
	case HS384:
		if len(ctx.key)*8 < 384 {
			return nil, ErrInvalidKeySize
		}
		hash = sha512.New384
	case HS512:
		if len(ctx.key)*8 < 512 {
			return nil, ErrInvalidKeySize
		}
		hash = sha512.New
	default:
		return nil, ErrUnsupportedAlgorithm
	}

	hmac := hmac.New(hash, ctx.key)

	// According to documentation, Write() on hash never fails
	_, _ = hmac.Write(payload)
	return hmac.Sum(nil), nil
}
