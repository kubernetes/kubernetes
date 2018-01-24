package auth

import (
	"testing"
)

func TestHeaderParsing(t *testing.T) {
	// Basic admin:password
	authorization := "Basic YWRtaW46cGFzc3dvcmQ="

	auth, err := parseAuthHeader(authorization)
	if err != nil {
		t.Error(err)
	}

	if auth.Name != "admin" {
		t.Errorf("Detected name does not match: '%s'", auth.Name)
	}
	if auth.Pass != "password" {
		t.Errorf("Detected password does not match: '%s'", auth.Pass)
	}
}

func TestEmptyHeader(t *testing.T) {
	if _, err := parseAuthHeader(""); err == nil {
		t.Errorf("Empty headers should generate errors")
	}
}
