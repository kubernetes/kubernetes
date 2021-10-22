package s3crypto

import (
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kms"
)

func TestAESGCMContentCipherBuilder(t *testing.T) {
	generator := mockGenerator{}
	if builder := AESGCMContentCipherBuilder(generator); builder == nil {
		t.Error("expected non-nil value")
	}
}

func TestAESGCMContentCipherNewEncryptor(t *testing.T) {
	generator := mockGenerator{}
	builder := AESGCMContentCipherBuilder(generator)
	cipher, err := builder.ContentCipher()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if cipher == nil {
		t.Errorf("expected non-nil vaue")
	}
}

func TestAESGCMContentCipherBuilderV2(t *testing.T) {
	builder := AESGCMContentCipherBuilderV2(mockGeneratorV2{})
	cipher, err := builder.ContentCipher()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if cipher == nil {
		t.Errorf("expected non-nil vaue")
	}
}

func TestGcmContentCipherBuilder_isFixtureEncryptionCompatible(t *testing.T) {
	builder := AESGCMContentCipherBuilder(NewKMSKeyGenerator(mockKMS{}, "cmkID"))
	features, ok := builder.(compatibleEncryptionFixture)
	if !ok {
		t.Errorf("expected to implement compatibleEncryptionFixture interface")
	}

	if err := features.isEncryptionVersionCompatible(v1ClientVersion); err != nil {
		t.Errorf("expected to receive no error, got %v", err)
	}

	if err := features.isEncryptionVersionCompatible(v2ClientVersion); err == nil {
		t.Errorf("expected to receive error, got nil")
	}
}

func TestGcmContentCipherBuilderV2_isFixtureEncryptionCompatible(t *testing.T) {
	builder := AESGCMContentCipherBuilderV2(NewKMSContextKeyGenerator(mockKMS{}, "cmkID", nil))
	features, ok := builder.(compatibleEncryptionFixture)
	if !ok {
		t.Errorf("expected to implement compatibleEncryptionFixture interface")
	}

	if err := features.isEncryptionVersionCompatible(v1ClientVersion); err == nil {
		t.Error("expected to receive error, got nil")
	}

	if err := features.isEncryptionVersionCompatible(v2ClientVersion); err != nil {
		t.Errorf("expected to receive no error, got %v", err)
	}
}

func TestRegisterAESGCMContentCipher(t *testing.T) {
	cr := NewCryptoRegistry()
	err := RegisterAESGCMContentCipher(cr)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if v, ok := cr.GetCEK("AES/GCM/NoPadding"); !ok {
		t.Fatal("expected cek handler to be registered")
	} else if v == nil {
		t.Fatal("expected non-nil cek handler")
	}

	if v, ok := cr.GetPadder("NoPadding"); !ok {
		t.Fatal("expected padder to be registered")
	} else if v != NoPadder {
		t.Fatal("padder did not match expected type")
	}

	err = RegisterAESGCMContentCipher(cr)
	if err == nil {
		t.Fatal("expected error, got none")
	} else if !strings.Contains(err.Error(), "duplicate cek registry entry") {
		t.Errorf("expected duplicate entry, got %v", err)
	}

	if _, ok := cr.RemoveCEK("AES/GCM/NoPadding"); !ok {
		t.Error("expected value to be removed")
	}
	err = RegisterAESGCMContentCipher(cr)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if _, ok := cr.RemoveCEK("AES/GCM/NoPadding"); !ok {
		t.Fatalf("expected value to be removed")
	}
	if _, ok := cr.RemovePadder("NoPadding"); !ok {
		t.Fatalf("expected value to be removed")
	}
	if err := cr.AddPadder("NoPadding", mockPadder{}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	err = RegisterAESGCMContentCipher(cr)
	if err == nil {
		t.Fatalf("expected error, got %v", err)
	} else if !strings.Contains(err.Error(), "does not match expected type") {
		t.Errorf("expected padder type error, got %v", err)
	}
}

func TestAESGCMContentCipherBuilderV2_isAWSFixture(t *testing.T) {
	builder := AESGCMContentCipherBuilderV2(NewKMSContextKeyGenerator(kms.New(unit.Session.Copy()), "cmk", nil))
	if !builder.(awsFixture).isAWSFixture() {
		t.Error("expected to be AWS ContentCipherBuilder constructed with a AWS CipherDataGenerator")
	}

	builder = AESGCMContentCipherBuilderV2(mockGeneratorV2{})
	if builder.(awsFixture).isAWSFixture() {
		t.Error("expected to return that this is not an AWS fixture")
	}
}
