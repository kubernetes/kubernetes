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
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"hash"
)

const (
	nonceBytes = 16
)

// NewCBCHMAC instantiates a new AEAD based on CBC+HMAC.
func NewCBCHMAC(key []byte, newBlockCipher func([]byte) (cipher.Block, error)) (cipher.AEAD, error) {
	keySize := len(key) / 2
	integrityKey := key[:keySize]
	encryptionKey := key[keySize:]

	blockCipher, err := newBlockCipher(encryptionKey)
	if err != nil {
		return nil, err
	}

	var hash func() hash.Hash
	switch keySize {
	case 16:
		hash = sha256.New
	case 24:
		hash = sha512.New384
	case 32:
		hash = sha512.New
	}

	return &cbcAEAD{
		hash:         hash,
		blockCipher:  blockCipher,
		authtagBytes: keySize,
		integrityKey: integrityKey,
	}, nil
}

// An AEAD based on CBC+HMAC
type cbcAEAD struct {
	hash         func() hash.Hash
	authtagBytes int
	integrityKey []byte
	blockCipher  cipher.Block
}

func (ctx *cbcAEAD) NonceSize() int {
	return nonceBytes
}

func (ctx *cbcAEAD) Overhead() int {
	// Maximum overhead is block size (for padding) plus auth tag length, where
	// the length of the auth tag is equivalent to the key size.
	return ctx.blockCipher.BlockSize() + ctx.authtagBytes
}

// Seal encrypts and authenticates the plaintext.
func (ctx *cbcAEAD) Seal(dst, nonce, plaintext, data []byte) []byte {
	// Output buffer -- must take care not to mangle plaintext input.
	ciphertext := make([]byte, uint64(len(plaintext))+uint64(ctx.Overhead()))[:len(plaintext)]
	copy(ciphertext, plaintext)
	ciphertext = padBuffer(ciphertext, ctx.blockCipher.BlockSize())

	cbc := cipher.NewCBCEncrypter(ctx.blockCipher, nonce)

	cbc.CryptBlocks(ciphertext, ciphertext)
	authtag := ctx.computeAuthTag(data, nonce, ciphertext)

	ret, out := resize(dst, uint64(len(dst))+uint64(len(ciphertext))+uint64(len(authtag)))
	copy(out, ciphertext)
	copy(out[len(ciphertext):], authtag)

	return ret
}

// Open decrypts and authenticates the ciphertext.
func (ctx *cbcAEAD) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(ciphertext) < ctx.authtagBytes {
		return nil, errors.New("go-jose/go-jose: invalid ciphertext (too short)")
	}

	offset := len(ciphertext) - ctx.authtagBytes
	expectedTag := ctx.computeAuthTag(data, nonce, ciphertext[:offset])
	match := subtle.ConstantTimeCompare(expectedTag, ciphertext[offset:])
	if match != 1 {
		return nil, errors.New("go-jose/go-jose: invalid ciphertext (auth tag mismatch)")
	}

	cbc := cipher.NewCBCDecrypter(ctx.blockCipher, nonce)

	// Make copy of ciphertext buffer, don't want to modify in place
	buffer := append([]byte{}, []byte(ciphertext[:offset])...)

	if len(buffer)%ctx.blockCipher.BlockSize() > 0 {
		return nil, errors.New("go-jose/go-jose: invalid ciphertext (invalid length)")
	}

	cbc.CryptBlocks(buffer, buffer)

	// Remove padding
	plaintext, err := unpadBuffer(buffer, ctx.blockCipher.BlockSize())
	if err != nil {
		return nil, err
	}

	ret, out := resize(dst, uint64(len(dst))+uint64(len(plaintext)))
	copy(out, plaintext)

	return ret, nil
}

// Compute an authentication tag
func (ctx *cbcAEAD) computeAuthTag(aad, nonce, ciphertext []byte) []byte {
	buffer := make([]byte, uint64(len(aad))+uint64(len(nonce))+uint64(len(ciphertext))+8)
	n := 0
	n += copy(buffer, aad)
	n += copy(buffer[n:], nonce)
	n += copy(buffer[n:], ciphertext)
	binary.BigEndian.PutUint64(buffer[n:], uint64(len(aad))*8)

	// According to documentation, Write() on hash.Hash never fails.
	hmac := hmac.New(ctx.hash, ctx.integrityKey)
	_, _ = hmac.Write(buffer)

	return hmac.Sum(nil)[:ctx.authtagBytes]
}

// resize ensures that the given slice has a capacity of at least n bytes.
// If the capacity of the slice is less than n, a new slice is allocated
// and the existing data will be copied.
func resize(in []byte, n uint64) (head, tail []byte) {
	if uint64(cap(in)) >= n {
		head = in[:n]
	} else {
		head = make([]byte, n)
		copy(head, in)
	}

	tail = head[len(in):]
	return
}

// Apply padding
func padBuffer(buffer []byte, blockSize int) []byte {
	missing := blockSize - (len(buffer) % blockSize)
	ret, out := resize(buffer, uint64(len(buffer))+uint64(missing))
	padding := bytes.Repeat([]byte{byte(missing)}, missing)
	copy(out, padding)
	return ret
}

// Remove padding
func unpadBuffer(buffer []byte, blockSize int) ([]byte, error) {
	if len(buffer)%blockSize != 0 {
		return nil, errors.New("go-jose/go-jose: invalid padding")
	}

	last := buffer[len(buffer)-1]
	count := int(last)

	if count == 0 || count > blockSize || count > len(buffer) {
		return nil, errors.New("go-jose/go-jose: invalid padding")
	}

	padding := bytes.Repeat([]byte{last}, count)
	if !bytes.HasSuffix(buffer, padding) {
		return nil, errors.New("go-jose/go-jose: invalid padding")
	}

	return buffer[:len(buffer)-count], nil
}
