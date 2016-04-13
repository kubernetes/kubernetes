package api

import (
	"testing"
)

func TestJsonContentType(t *testing.T) {
	if !MatchesContentType("application/json", "application/json") {
		t.Fail()
	}

	if !MatchesContentType("application/json; charset=utf-8", "application/json") {
		t.Fail()
	}

	if MatchesContentType("dockerapplication/json", "application/json") {
		t.Fail()
	}
}
