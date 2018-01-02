package s3crypto

import (
	"io"
)

const (
	gcmKeySize   = 32
	gcmNonceSize = 12
)

type gcmContentCipherBuilder struct {
	generator CipherDataGenerator
}

// AESGCMContentCipherBuilder returns a new encryption only mode structure with a specific cipher
// for the master key
func AESGCMContentCipherBuilder(generator CipherDataGenerator) ContentCipherBuilder {
	return gcmContentCipherBuilder{generator}
}

func (builder gcmContentCipherBuilder) ContentCipher() (ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherData(gcmKeySize, gcmNonceSize)
	if err != nil {
		return nil, err
	}

	return newAESGCMContentCipher(cd)
}

func newAESGCMContentCipher(cd CipherData) (ContentCipher, error) {
	cd.CEKAlgorithm = AESGCMNoPadding
	cd.TagLength = "128"

	cipher, err := newAESGCM(cd)
	if err != nil {
		return nil, err
	}

	return &aesGCMContentCipher{
		CipherData: cd,
		Cipher:     cipher,
	}, nil
}

// AESGCMContentCipher will use AES GCM for the main cipher.
type aesGCMContentCipher struct {
	CipherData CipherData
	Cipher     Cipher
}

// EncryptContents will generate a random key and iv and encrypt the data using cbc
func (cc *aesGCMContentCipher) EncryptContents(src io.Reader) (io.Reader, error) {
	return cc.Cipher.Encrypt(src), nil
}

// DecryptContents will use the symmetric key provider to instantiate a new GCM cipher.
// We grab a decrypt reader from gcm and wrap it in a CryptoReadCloser. The only error
// expected here is when the key or iv is of invalid length.
func (cc *aesGCMContentCipher) DecryptContents(src io.ReadCloser) (io.ReadCloser, error) {
	reader := cc.Cipher.Decrypt(src)
	return &CryptoReadCloser{Body: src, Decrypter: reader}, nil
}

// GetCipherData returns cipher data
func (cc aesGCMContentCipher) GetCipherData() CipherData {
	return cc.CipherData
}
