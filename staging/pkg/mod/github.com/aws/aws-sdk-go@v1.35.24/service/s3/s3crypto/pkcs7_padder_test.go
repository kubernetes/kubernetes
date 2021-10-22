package s3crypto_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/service/s3/s3crypto"
)

func padTest(size int, t *testing.T) {
	padder := s3crypto.NewPKCS7Padder(size)
	for i := 0; i < size; i++ {
		input := make([]byte, i)
		expected := append(input, bytes.Repeat([]byte{byte(size - i)}, size-i)...)
		b, err := padder.Pad(input, len(input))
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

func unpadTest(size int, t *testing.T) {
	padder := s3crypto.NewPKCS7Padder(size)
	for i := 0; i < size; i++ {
		expected := make([]byte, i)
		input := append(expected, bytes.Repeat([]byte{byte(size - i)}, size-i)...)
		b, err := padder.Unpad(input)
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

func TestPKCS7Padding(t *testing.T) {
	padTest(10, t)
	padTest(16, t)
	padTest(255, t)
}

func TestPKCS7Unpadding(t *testing.T) {
	unpadTest(10, t)
	unpadTest(16, t)
	unpadTest(255, t)
}
