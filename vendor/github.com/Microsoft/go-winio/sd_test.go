package winio

import "testing"

func TestLookupInvalidSid(t *testing.T) {
	_, err := LookupSidByName(".\\weoifjdsklfj")
	aerr, ok := err.(*AccountLookupError)
	if !ok || aerr.Err != cERROR_NONE_MAPPED {
		t.Fatalf("expected AccountLookupError with ERROR_NONE_MAPPED, got %s", err)
	}
}

func TestLookupValidSid(t *testing.T) {
	sid, err := LookupSidByName("Everyone")
	if err != nil || sid != "S-1-1-0" {
		t.Fatal("expected S-1-1-0, got %s, %s", sid, err)
	}
}

func TestLookupEmptyNameFails(t *testing.T) {
	_, err := LookupSidByName(".\\weoifjdsklfj")
	aerr, ok := err.(*AccountLookupError)
	if !ok || aerr.Err != cERROR_NONE_MAPPED {
		t.Fatalf("expected AccountLookupError with ERROR_NONE_MAPPED, got %s", err)
	}
}
