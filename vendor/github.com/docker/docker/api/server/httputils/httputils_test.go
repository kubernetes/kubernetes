package httputils

import "testing"

// matchesContentType
func TestJsonContentType(t *testing.T) {
	if !matchesContentType("application/json", "application/json") {
		t.Fail()
	}

	if !matchesContentType("application/json; charset=utf-8", "application/json") {
		t.Fail()
	}

	if matchesContentType("dockerapplication/json", "application/json") {
		t.Fail()
	}
}
