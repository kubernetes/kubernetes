package toml

import (
	"fmt"
	"testing"
)

func testResult(t *testing.T, key string, expected []string) {
	parsed, err := parseKey(key)
	if err != nil {
		t.Fatal("Unexpected error:", err)
	}
	if len(expected) != len(parsed) {
		t.Fatal("Expected length", len(expected), "but", len(parsed), "parsed")
	}
	for index, expectedKey := range expected {
		if expectedKey != parsed[index] {
			t.Fatal("Expected", expectedKey, "at index", index, "but found", parsed[index])
		}
	}
}

func testError(t *testing.T, key string, expectedError string) {
	_, err := parseKey(key)
	if fmt.Sprintf("%s", err) != expectedError {
		t.Fatalf("Expected error \"%s\", but got \"%s\".", expectedError, err)
	}
}

func TestBareKeyBasic(t *testing.T) {
	testResult(t, "test", []string{"test"})
}

func TestBareKeyDotted(t *testing.T) {
	testResult(t, "this.is.a.key", []string{"this", "is", "a", "key"})
}

func TestDottedKeyBasic(t *testing.T) {
	testResult(t, "\"a.dotted.key\"", []string{"a.dotted.key"})
}

func TestBaseKeyPound(t *testing.T) {
	testError(t, "hello#world", "invalid bare character: #")
}

func TestEmptyKey(t *testing.T) {
	testError(t, "", "empty key")
	testError(t, " ", "empty key")
}
