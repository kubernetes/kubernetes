package s3crypto_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestAESGCMContentCipherBuilder(t *testing.T) {
	generator := mockGenerator{}
	if builder := s3crypto.AESGCMContentCipherBuilder(generator); builder == nil {
		t.Error("expected non-nil value")
	}
}

func TestAESGCMContentCipherNewEncryptor(t *testing.T) {
	generator := mockGenerator{}
	builder := s3crypto.AESGCMContentCipherBuilder(generator)
	cipher, err := builder.ContentCipher()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if cipher == nil {
		t.Errorf("expected non-nil vaue")
	}
}
