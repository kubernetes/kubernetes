package nat

import (
	"strings"
	"testing"
)

func TestParsePortRange(t *testing.T) {
	if start, end, err := ParsePortRange("8000-8080"); err != nil || start != 8000 || end != 8080 {
		t.Fatalf("Error: %s or Expecting {start,end} values {8000,8080} but found {%d,%d}.", err, start, end)
	}
}

func TestParsePortRangeEmpty(t *testing.T) {
	if _, _, err := ParsePortRange(""); err == nil || err.Error() != "Empty string specified for ports." {
		t.Fatalf("Expected error 'Empty string specified for ports.', got %v", err)
	}
}

func TestParsePortRangeWithNoRange(t *testing.T) {
	start, end, err := ParsePortRange("8080")
	if err != nil {
		t.Fatal(err)
	}
	if start != 8080 || end != 8080 {
		t.Fatalf("Expected start and end to be the same and equal to 8080, but were %v and %v", start, end)
	}
}

func TestParsePortRangeIncorrectRange(t *testing.T) {
	if _, _, err := ParsePortRange("9000-8080"); err == nil || !strings.Contains(err.Error(), "Invalid range specified for the Port") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}

func TestParsePortRangeIncorrectEndRange(t *testing.T) {
	if _, _, err := ParsePortRange("8000-a"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}

	if _, _, err := ParsePortRange("8000-30a"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}

func TestParsePortRangeIncorrectStartRange(t *testing.T) {
	if _, _, err := ParsePortRange("a-8000"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}

	if _, _, err := ParsePortRange("30a-8000"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}
