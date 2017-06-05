package cryptoutil

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
)

// pad uses the PKCS#7 padding scheme to align the a payload to a specific block size
func pad(plaintext []byte, bsize int) ([]byte, error) {
	if bsize >= 256 {
		return nil, errors.New("bsize must be < 256")
	}
	pad := bsize - (len(plaintext) % bsize)
	if pad == 0 {
		pad = bsize
	}
	for i := 0; i < pad; i++ {
		plaintext = append(plaintext, byte(pad))
	}
	return plaintext, nil
}

// unpad strips the padding previously added using the PKCS#7 padding scheme
func unpad(paddedtext []byte) ([]byte, error) {
	length := len(paddedtext)
	paddedtext, lbyte := paddedtext[:length-1], paddedtext[length-1]
	pad := int(lbyte)
	if pad >= 256 || pad > length {
		return nil, errors.New("padding malformed")
	}
	return paddedtext[:length-(pad)], nil
}

// AESEncrypt encrypts a payload with an AES cipher.
// The returned ciphertext has three notable properties:
// 1. ciphertext is aligned to the standard AES block size
// 2. ciphertext is padded using PKCS#7
// 3. IV is prepended to the ciphertext
func AESEncrypt(plaintext, key []byte) ([]byte, error) {
	plaintext, err := pad(plaintext, aes.BlockSize)
	if err != nil {
		return nil, err
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	ciphertext := make([]byte, aes.BlockSize+len(plaintext))
	iv := ciphertext[:aes.BlockSize]
	if _, err := rand.Read(iv); err != nil {
		return nil, err
	}

	mode := cipher.NewCBCEncrypter(block, iv)
	mode.CryptBlocks(ciphertext[aes.BlockSize:], plaintext)

	return ciphertext, nil
}

// AESDecrypt decrypts an encrypted payload with an AES cipher.
// The decryption algorithm makes three assumptions:
// 1. ciphertext is aligned to the standard AES block size
// 2. ciphertext is padded using PKCS#7
// 3. the IV is prepended to ciphertext
func AESDecrypt(ciphertext, key []byte) ([]byte, error) {
	if len(ciphertext) < aes.BlockSize {
		return nil, errors.New("ciphertext too short")
	}

	iv := ciphertext[:aes.BlockSize]
	ciphertext = ciphertext[aes.BlockSize:]

	if len(ciphertext)%aes.BlockSize != 0 {
		return nil, errors.New("ciphertext is not a multiple of the block size")
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	mode := cipher.NewCBCDecrypter(block, iv)
	mode.CryptBlocks(ciphertext, ciphertext)

	if len(ciphertext)%aes.BlockSize != 0 {
		return nil, errors.New("ciphertext is not a multiple of the block size")
	}

	return unpad(ciphertext)
}
