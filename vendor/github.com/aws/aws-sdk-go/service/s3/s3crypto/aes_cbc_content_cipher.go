package s3crypto

import (
	"io"
	"strings"
)

const (
	cbcKeySize   = 32
	cbcNonceSize = 16
)

type cbcContentCipherBuilder struct {
	generator CipherDataGenerator
	padder    Padder
}

// AESCBCContentCipherBuilder returns a new encryption only mode structure with a specific cipher
// for the master key
func AESCBCContentCipherBuilder(generator CipherDataGenerator, padder Padder) ContentCipherBuilder {
	return cbcContentCipherBuilder{generator: generator, padder: padder}
}

func (builder cbcContentCipherBuilder) ContentCipher() (ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherData(cbcKeySize, cbcNonceSize)
	if err != nil {
		return nil, err
	}

	cd.Padder = builder.padder
	return newAESCBCContentCipher(cd)
}

// newAESCBCContentCipher will create a new aes cbc content cipher. If the cipher data's
// will set the CEK algorithm if it hasn't been set.
func newAESCBCContentCipher(cd CipherData) (ContentCipher, error) {
	if len(cd.CEKAlgorithm) == 0 {
		cd.CEKAlgorithm = strings.Join([]string{AESCBC, cd.Padder.Name()}, "/")
	}
	cipher, err := newAESCBC(cd, cd.Padder)
	if err != nil {
		return nil, err
	}

	return &aesCBCContentCipher{
		CipherData: cd,
		Cipher:     cipher,
	}, nil
}

// aesCBCContentCipher will use AES CBC for the main cipher.
type aesCBCContentCipher struct {
	CipherData CipherData
	Cipher     Cipher
}

// EncryptContents will generate a random key and iv and encrypt the data using cbc
func (cc *aesCBCContentCipher) EncryptContents(src io.Reader) (io.Reader, error) {
	return cc.Cipher.Encrypt(src), nil
}

// DecryptContents will use the symmetric key provider to instantiate a new CBC cipher.
// We grab a decrypt reader from CBC and wrap it in a CryptoReadCloser. The only error
// expected here is when the key or iv is of invalid length.
func (cc *aesCBCContentCipher) DecryptContents(src io.ReadCloser) (io.ReadCloser, error) {
	reader := cc.Cipher.Decrypt(src)
	return &CryptoReadCloser{Body: src, Decrypter: reader}, nil
}

// GetCipherData returns cipher data
func (cc aesCBCContentCipher) GetCipherData() CipherData {
	return cc.CipherData
}
