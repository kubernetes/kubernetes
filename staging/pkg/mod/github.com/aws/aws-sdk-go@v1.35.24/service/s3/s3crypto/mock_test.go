package s3crypto

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/kms/kmsiface"
)

type mockGenerator struct{}

func (m mockGenerator) GenerateCipherData(keySize, ivSize int) (CipherData, error) {
	cd := CipherData{
		Key: make([]byte, keySize),
		IV:  make([]byte, ivSize),
	}
	return cd, nil
}

func (m mockGenerator) DecryptKey(key []byte) ([]byte, error) {
	return make([]byte, 16), nil
}

type mockGeneratorV2 struct{}

func (m mockGeneratorV2) GenerateCipherDataWithCEKAlg(ctx aws.Context, keySize int, ivSize int, cekAlg string) (CipherData, error) {
	cd := CipherData{
		Key: make([]byte, keySize),
		IV:  make([]byte, ivSize),
	}
	return cd, nil
}

func (m mockGeneratorV2) DecryptKey(key []byte) ([]byte, error) {
	return make([]byte, 16), nil
}

func (m mockGeneratorV2) isEncryptionVersionCompatible(version clientVersion) error {
	if version != v2ClientVersion {
		return fmt.Errorf("mock error about version")
	}
	return nil
}

type mockCipherBuilder struct {
	generator CipherDataGenerator
}

func (builder mockCipherBuilder) isEncryptionVersionCompatible(version clientVersion) error {
	if version != v1ClientVersion {
		return fmt.Errorf("mock error about version")
	}
	return nil
}

func (builder mockCipherBuilder) ContentCipher() (ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherData(32, 16)
	if err != nil {
		return nil, err
	}
	return &mockContentCipher{cd}, nil
}

type mockCipherBuilderV2 struct {
	generator CipherDataGeneratorWithCEKAlg
}

func (builder mockCipherBuilderV2) isEncryptionVersionCompatible(version clientVersion) error {
	if version != v2ClientVersion {
		return fmt.Errorf("mock error about version")
	}
	return nil
}

func (builder mockCipherBuilderV2) ContentCipher() (ContentCipher, error) {
	cd, err := builder.generator.GenerateCipherDataWithCEKAlg(aws.BackgroundContext(), 32, 16, "mock-cek-alg")
	if err != nil {
		return nil, err
	}
	return &mockContentCipher{cd}, nil
}

type mockContentCipher struct {
	cd CipherData
}

func (cipher *mockContentCipher) GetCipherData() CipherData {
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

type mockKMS struct {
	kmsiface.KMSAPI
}

type mockPadder struct {
}

func (m mockPadder) Pad(i []byte, i2 int) ([]byte, error) {
	return i, nil
}

func (m mockPadder) Unpad(i []byte) ([]byte, error) {
	return i, nil
}

func (m mockPadder) Name() string {
	return "mockPadder"
}
