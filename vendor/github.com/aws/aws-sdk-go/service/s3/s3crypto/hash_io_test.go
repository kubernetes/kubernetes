package s3crypto

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// From Go stdlib encoding/sha256 test cases
func TestSHA256(t *testing.T) {
	sha := newSHA256Writer(nil)
	expected, _ := hex.DecodeString("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
	b := sha.GetValue()

	if !bytes.Equal(expected, b) {
		t.Errorf("expected equivalent sha values, but received otherwise")
	}
}

func TestSHA256_Case2(t *testing.T) {
	sha := newSHA256Writer(bytes.NewBuffer([]byte{}))
	sha.Write([]byte("hello"))
	expected, _ := hex.DecodeString("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
	b := sha.GetValue()

	if !bytes.Equal(expected, b) {
		t.Errorf("expected equivalent sha values, but received otherwise")
	}
}
