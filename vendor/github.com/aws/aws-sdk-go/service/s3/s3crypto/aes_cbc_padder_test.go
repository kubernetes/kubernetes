package s3crypto

import (
	"bytes"
	"fmt"
	"testing"
)

func TestAESCBCPadding(t *testing.T) {
	for i := 0; i < 16; i++ {
		input := make([]byte, i)
		expected := append(input, bytes.Repeat([]byte{byte(16 - i)}, 16-i)...)
		b, err := AESCBCPadder.Pad(input, len(input))
		if err != nil {
			t.Fatal("Expected error to be nil but received " + err.Error())
		}
		if len(b) != len(expected) {
			t.Fatal(fmt.Sprintf("Case %d: data is not of the same length", i))
		}
		if bytes.Compare(b, expected) != 0 {
			t.Fatal(fmt.Sprintf("Expected %v but got %v", expected, b))
		}
	}
}

func TestAESCBCUnpadding(t *testing.T) {
	for i := 0; i < 16; i++ {
		expected := make([]byte, i)
		input := append(expected, bytes.Repeat([]byte{byte(16 - i)}, 16-i)...)
		b, err := AESCBCPadder.Unpad(input)
		if err != nil {
			t.Fatal("Error received, was expecting nil: " + err.Error())
		}
		if len(b) != len(expected) {
			t.Fatal(fmt.Sprintf("Case %d: data is not of the same length", i))
		}
		if bytes.Compare(b, expected) != 0 {
			t.Fatal(fmt.Sprintf("Expected %v but got %v", expected, b))
		}
	}
}
