package cryptoutil

import (
	"reflect"
	"testing"
)

func TestPadUnpad(t *testing.T) {
	tests := []struct {
		plaintext []byte
		bsize     int
		padded    []byte
	}{
		{
			plaintext: []byte{1, 2, 3, 4},
			bsize:     7,
			padded:    []byte{1, 2, 3, 4, 3, 3, 3},
		},
		{
			plaintext: []byte{1, 2, 3, 4, 5, 6, 7},
			bsize:     3,
			padded:    []byte{1, 2, 3, 4, 5, 6, 7, 2, 2},
		},
		{
			plaintext: []byte{9, 9, 9, 9},
			bsize:     4,
			padded:    []byte{9, 9, 9, 9, 4, 4, 4, 4},
		},
	}

	for i, tt := range tests {
		padded, err := pad(tt.plaintext, tt.bsize)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(tt.padded, padded) {
			t.Errorf("case %d: want=%v got=%v", i, tt.padded, padded)
			continue
		}

		plaintext, err := unpad(tt.padded)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(tt.plaintext, plaintext) {
			t.Errorf("case %d: want=%v got=%v", i, tt.plaintext, plaintext)
			continue
		}
	}
}

func TestPadMaxBlockSize(t *testing.T) {
	_, err := pad([]byte{1, 2, 3}, 256)
	if err == nil {
		t.Errorf("Expected non-nil error")
	}
}

func TestAESEncryptDecrypt(t *testing.T) {
	message := []byte("Let me worry about blank.")
	key := append([]byte("shark"), make([]byte, 27)...)

	ciphertext, err := AESEncrypt(message, key)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if reflect.DeepEqual(message, ciphertext) {
		t.Fatal("Encrypted data matches original payload")
	}

	decrypted, err := AESDecrypt(ciphertext, key)
	if !reflect.DeepEqual(message, decrypted) {
		t.Fatalf("Decrypted data does not match original payload: want=%v got=%v", message, decrypted)
	}
}

func TestAESDecryptWrongKey(t *testing.T) {
	message := []byte("My bones!")
	key := append([]byte("shark"), make([]byte, 27)...)

	ciphertext, err := AESEncrypt(message, key)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	wrongKey := append([]byte("sheep"), make([]byte, 27)...)
	decrypted, _ := AESDecrypt(ciphertext, wrongKey)
	if reflect.DeepEqual(message, decrypted) {
		t.Fatalf("Data decrypted with different key matches original payload")
	}
}
