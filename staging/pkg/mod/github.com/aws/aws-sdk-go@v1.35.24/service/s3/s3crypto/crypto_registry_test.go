package s3crypto

import (
	"strings"
	"testing"
)

func TestCryptoRegistry_Wrap(t *testing.T) {
	cr := NewCryptoRegistry()

	mockWrap := WrapEntry(func(envelope Envelope) (CipherDataDecrypter, error) {
		return nil, nil
	})

	if _, ok := cr.GetWrap("foo"); ok {
		t.Errorf("expected wrapper to not be present")
	}

	if _, ok := cr.RemoveWrap("foo"); ok {
		t.Errorf("expected wrapped to not have been removed")
	}

	if err := cr.AddWrap("foo", nil); err == nil {
		t.Errorf("expected error, got none")
	}

	if err := cr.AddWrap("foo", mockWrap); err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if err := cr.AddWrap("foo", mockWrap); err == nil {
		t.Error("expected error, got none")
	}

	if v, ok := cr.GetWrap("foo"); !ok || v == nil {
		t.Error("expected wrapper to be present and not nil")
	}

	if v, ok := cr.RemoveWrap("foo"); !ok || v == nil {
		t.Error("expected wrapper to have been removed and not nil")
	}

	if _, ok := cr.GetWrap("foo"); ok {
		t.Error("expected wrapper to have been removed and not nil")
	}
}

func TestCryptoRegistry_CEK(t *testing.T) {
	cr := NewCryptoRegistry()

	mockEntry := CEKEntry(func(data CipherData) (ContentCipher, error) {
		return nil, nil
	})

	if _, ok := cr.GetCEK("foo"); ok {
		t.Errorf("expected wrapper to not be present")
	}

	if _, ok := cr.RemoveCEK("foo"); ok {
		t.Errorf("expected wrapped to not have been removed")
	}

	if err := cr.AddCEK("foo", nil); err == nil {
		t.Errorf("expected error, got none")
	}

	if err := cr.AddCEK("foo", mockEntry); err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if err := cr.AddCEK("foo", mockEntry); err == nil {
		t.Error("expected error, got none")
	}

	if v, ok := cr.GetCEK("foo"); !ok || v == nil {
		t.Error("expected wrapper to be present and not nil")
	}

	if v, ok := cr.RemoveCEK("foo"); !ok || v == nil {
		t.Error("expected wrapper to have been removed and not nil")
	}

	if _, ok := cr.GetCEK("foo"); ok {
		t.Error("expected wrapper to have been removed and not nil")
	}
}

func TestCryptoRegistry_Padder(t *testing.T) {
	cr := NewCryptoRegistry()

	padder := &mockPadder{}

	if _, ok := cr.GetPadder("foo"); ok {
		t.Errorf("expected wrapper to not be present")
	}

	if _, ok := cr.RemovePadder("foo"); ok {
		t.Errorf("expected wrapped to not have been removed")
	}

	if err := cr.AddPadder("foo", nil); err == nil {
		t.Errorf("expected error, got none")
	}

	if err := cr.AddPadder("foo", padder); err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if err := cr.AddPadder("foo", padder); err == nil {
		t.Error("expected error, got none")
	}

	if v, ok := cr.GetPadder("foo"); !ok || v == nil {
		t.Error("expected wrapper to be present and not nil")
	}

	if v, ok := cr.RemovePadder("foo"); !ok || v == nil {
		t.Error("expected wrapper to have been removed and not nil")
	}
}

func TestCryptoRegistry_valid(t *testing.T) {
	cr := NewCryptoRegistry()

	if err := cr.valid(); err == nil {
		t.Errorf("expected error, got none")
	} else if e, a := "at least one key wrapping algorithms must be provided", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := cr.AddWrap("foo", func(envelope Envelope) (CipherDataDecrypter, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err := cr.valid(); err == nil {
		t.Fatalf("expected error, got none")
	} else if e, a := "least one content decryption algorithms must be provided", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := cr.AddCEK("foo", func(data CipherData) (ContentCipher, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err := cr.valid(); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}
