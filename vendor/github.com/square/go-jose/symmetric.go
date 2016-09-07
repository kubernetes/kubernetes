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
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/subtle"
	"errors"
	"hash"
	"io"

	"github.com/square/go-jose/cipher"
)

// Random reader (stubbed out in tests)
var randReader = rand.Reader

// Dummy key cipher for shared symmetric key mode
type symmetricKeyCipher struct {
	key []byte // Pre-shared content-encryption key
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
		authtagBytes: 16,
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

// newSymmetricRecipient creates a JWE encrypter based on AES-GCM key wrap.
func newSymmetricRecipient(keyAlg KeyAlgorithm, key []byte) (recipientKeyInfo, error) {
	switch keyAlg {
	case DIRECT, A128GCMKW, A192GCMKW, A256GCMKW, A128KW, A192KW, A256KW:
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
	_, err := io.ReadFull(randReader, key)
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
	_, err = io.ReadFull(randReader, iv)
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

		return recipientInfo{
			header: &rawHeader{
				Iv:  newBuffer(parts.iv),
				Tag: newBuffer(parts.tag),
			},
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
	}

	return recipientInfo{}, ErrUnsupportedAlgorithm
}

// Decrypt the content encryption key.
func (ctx *symmetricKeyCipher) decryptKey(headers rawHeader, recipient *recipientInfo, generator keyGenerator) ([]byte, error) {
	switch KeyAlgorithm(headers.Alg) {
	case DIRECT:
		cek := make([]byte, len(ctx.key))
		copy(cek, ctx.key)
		return cek, nil
	case A128GCMKW, A192GCMKW, A256GCMKW:
		aead := newAESGCM(len(ctx.key))

		parts := &aeadParts{
			iv:         headers.Iv.bytes(),
			ciphertext: recipient.encryptedKey,
			tag:        headers.Tag.bytes(),
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
	}

	return nil, ErrUnsupportedAlgorithm
}

// Sign the given payload
func (ctx symmetricMac) signPayload(payload []byte, alg SignatureAlgorithm) (Signature, error) {
	mac, err := ctx.hmac(payload, alg)
	if err != nil {
		return Signature{}, errors.New("square/go-jose: failed to compute hmac")
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
		return errors.New("square/go-jose: failed to compute hmac")
	}

	if len(mac) != len(expected) {
		return errors.New("square/go-jose: invalid hmac")
	}

	match := subtle.ConstantTimeCompare(mac, expected)
	if match != 1 {
		return errors.New("square/go-jose: invalid hmac")
	}

	return nil
}

// Compute the HMAC based on the given alg value
func (ctx symmetricMac) hmac(payload []byte, alg SignatureAlgorithm) ([]byte, error) {
	var hash func() hash.Hash

	switch alg {
	case HS256:
		hash = sha256.New
	case HS384:
		hash = sha512.New384
	case HS512:
		hash = sha512.New
	default:
		return nil, ErrUnsupportedAlgorithm
	}

	hmac := hmac.New(hash, ctx.key)

	// According to documentation, Write() on hash never fails
	_, _ = hmac.Write(payload)
	return hmac.Sum(nil), nil
}
