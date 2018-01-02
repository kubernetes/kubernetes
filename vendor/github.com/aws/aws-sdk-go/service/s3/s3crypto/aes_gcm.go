package s3crypto

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"io"
	"io/ioutil"
)

// AESGCM Symmetric encryption algorithm. Since Golang designed this
// with only TLS in mind. We have to load it all into memory meaning
// this isn't streamed.
type aesGCM struct {
	aead  cipher.AEAD
	nonce []byte
}

// newAESGCM creates a new AES GCM cipher. Expects keys to be of
// the correct size.
//
// Example:
//
//	cd := &s3crypto.CipherData{
//		Key: key,
//		"IV": iv,
//	}
//	cipher, err := s3crypto.newAESGCM(cd)
func newAESGCM(cd CipherData) (Cipher, error) {
	block, err := aes.NewCipher(cd.Key)
	if err != nil {
		return nil, err
	}

	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	return &aesGCM{aesgcm, cd.IV}, nil
}

// Encrypt will encrypt the data using AES GCM
// Tag will be included as the last 16 bytes of the slice
func (c *aesGCM) Encrypt(src io.Reader) io.Reader {
	reader := &gcmEncryptReader{
		encrypter: c.aead,
		nonce:     c.nonce,
		src:       src,
	}
	return reader
}

type gcmEncryptReader struct {
	encrypter cipher.AEAD
	nonce     []byte
	src       io.Reader
	buf       *bytes.Buffer
}

func (reader *gcmEncryptReader) Read(data []byte) (int, error) {
	if reader.buf == nil {
		b, err := ioutil.ReadAll(reader.src)
		if err != nil {
			return len(b), err
		}
		b = reader.encrypter.Seal(b[:0], reader.nonce, b, nil)
		reader.buf = bytes.NewBuffer(b)
	}

	return reader.buf.Read(data)
}

// Decrypt will decrypt the data using AES GCM
func (c *aesGCM) Decrypt(src io.Reader) io.Reader {
	return &gcmDecryptReader{
		decrypter: c.aead,
		nonce:     c.nonce,
		src:       src,
	}
}

type gcmDecryptReader struct {
	decrypter cipher.AEAD
	nonce     []byte
	src       io.Reader
	buf       *bytes.Buffer
}

func (reader *gcmDecryptReader) Read(data []byte) (int, error) {
	if reader.buf == nil {
		b, err := ioutil.ReadAll(reader.src)
		if err != nil {
			return len(b), err
		}
		b, err = reader.decrypter.Open(b[:0], reader.nonce, b, nil)
		if err != nil {
			return len(b), err
		}

		reader.buf = bytes.NewBuffer(b)
	}

	return reader.buf.Read(data)
}
