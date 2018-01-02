package s3crypto_test

import (
	"bytes"
	"io"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

type mockGenerator struct {
}

func (m mockGenerator) GenerateCipherData(keySize, ivSize int) (s3crypto.CipherData, error) {
	cd := s3crypto.CipherData{
		Key: make([]byte, keySize),
		IV:  make([]byte, ivSize),
	}
	return cd, nil
}

func (m mockGenerator) EncryptKey(key []byte) ([]byte, error) {
	size := len(key)
	b := bytes.Repeat([]byte{1}, size)
	return b, nil
}

func (m mockGenerator) DecryptKey(key []byte) ([]byte, error) {
	return make([]byte, 16), nil

}

type mockCipherBuilder struct {
	generator s3crypto.CipherDataGenerator
}

func (builder mockCipherBuilder) ContentCipher() (s3crypto.ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherData(32, 16)
	if err != nil {
		return nil, err
	}
	return &mockContentCipher{cd}, nil
}

type mockContentCipher struct {
	cd s3crypto.CipherData
}

func (cipher *mockContentCipher) GetCipherData() s3crypto.CipherData {
	return cipher.cd
}

func (cipher *mockContentCipher) EncryptContents(src io.Reader) (io.Reader, error) {
	b, err := ioutil.ReadAll(src)
	if err != nil {
		return nil, err
	}
	size := len(b)
	b = bytes.Repeat([]byte{1}, size)
	return bytes.NewReader(b), nil
}

func (cipher *mockContentCipher) DecryptContents(src io.ReadCloser) (io.ReadCloser, error) {
	b, err := ioutil.ReadAll(src)
	if err != nil {
		return nil, err
	}
	size := len(b)
	return ioutil.NopCloser(bytes.NewReader(make([]byte, size))), nil
}
