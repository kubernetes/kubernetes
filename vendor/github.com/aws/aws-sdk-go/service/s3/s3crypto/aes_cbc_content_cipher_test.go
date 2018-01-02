package s3crypto_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func TestAESCBCBuilder(t *testing.T) {
	generator := mockGenerator{}
	builder := s3crypto.AESCBCContentCipherBuilder(generator, s3crypto.NoPadder)
	if builder == nil {
		t.Fatal(builder)
	}

	_, err := builder.ContentCipher()
	if err != nil {
		t.Fatal(err)
	}
}
