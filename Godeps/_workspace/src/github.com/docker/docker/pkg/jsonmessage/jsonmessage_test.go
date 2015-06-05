package jsonmessage

import (
	"testing"
)

func TestError(t *testing.T) {
	je := JSONError{404, "Not found"}
	if je.Error() != "Not found" {
		t.Fatalf("Expected 'Not found' got '%s'", je.Error())
	}
}

func TestProgress(t *testing.T) {
	jp := JSONProgress{}
	if jp.String() != "" {
		t.Fatalf("Expected empty string, got '%s'", jp.String())
	}

	expected := "     1 B"
	jp2 := JSONProgress{Current: 1}
	if jp2.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp2.String())
	}

	expected = "[=========================>                         ]     50 B/100 B"
	jp3 := JSONProgress{Current: 50, Total: 100}
	if jp3.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp3.String())
	}

	// this number can't be negetive gh#7136
	expected = "[==================================================>]     50 B/40 B"
	jp4 := JSONProgress{Current: 50, Total: 40}
	if jp4.String() != expected {
		t.Fatalf("Expected %q, got %q", expected, jp4.String())
	}
}
