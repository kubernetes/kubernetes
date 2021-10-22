package s3crypto

import (
	"strings"
	"testing"
)

func TestAESCBCBuilder(t *testing.T) {
	generator := mockGenerator{}
	builder := AESCBCContentCipherBuilder(generator, NoPadder)
	if builder == nil {
		t.Fatal(builder)
	}

	_, err := builder.ContentCipher()
	if err != nil {
		t.Fatal(err)
	}
}

func TestAesCBCContentCipher_isFixtureEncryptionCompatible(t *testing.T) {
	generator := mockGenerator{}
	builder := AESCBCContentCipherBuilder(generator, NoPadder)
	if builder == nil {
		t.Fatal("expected builder to not be nil")
	}

	compatibility, ok := builder.(compatibleEncryptionFixture)
	if !ok {
		t.Fatal("expected builder to implement compatibleEncryptionFixture interface")
	}

	if err := compatibility.isEncryptionVersionCompatible(v1ClientVersion); err != nil {
		t.Errorf("expected builder to be compatible with v1 client")
	}

	if err := compatibility.isEncryptionVersionCompatible(v2ClientVersion); err == nil {
		t.Errorf("expected builder to not be compatible with v2 client")
	}
}

func TestRegisterAESCBCContentCipher(t *testing.T) {
	cr := NewCryptoRegistry()
	padder := AESCBCPadder
	err := RegisterAESCBCContentCipher(cr, padder)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if v, ok := cr.GetCEK("AES/CBC/PKCS5Padding"); !ok {
		t.Fatal("expected cek algorithm handler to registered")
	} else if v == nil {
		t.Fatal("expected non-nil cek handler to be registered")
	}

	if v, ok := cr.GetPadder("AES/CBC/PKCS5Padding"); !ok {
		t.Fatal("expected padder to be registered")
	} else if v != padder {
		t.Fatal("padder did not match provided value")
	}

	// try to register padder again
	err = RegisterAESCBCContentCipher(cr, padder)
	if err == nil {
		t.Fatal("expected error, got none")
	} else if !strings.Contains(err.Error(), "duplicate cek registry entry") {
		t.Errorf("expected duplicate cek entry, got %v", err)
	}

	// try to regster padder with cek removed but padder entry still present
	if _, ok := cr.RemoveCEK("AES/CBC/PKCS5Padding"); !ok {
		t.Fatalf("expected value to be removed")
	}
	err = RegisterAESCBCContentCipher(cr, padder)
	if err == nil {
		t.Fatal("expected error, got none")
	} else if !strings.Contains(err.Error(), "duplicate padder registry entry") {
		t.Errorf("expected duplicate padder entry, got %v", err)
	}
}
