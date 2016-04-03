package useragent

import "testing"

func TestVersionInfo(t *testing.T) {
	vi := VersionInfo{"foo", "bar"}
	if !vi.isValid() {
		t.Fatalf("VersionInfo should be valid")
	}
	vi = VersionInfo{"", "bar"}
	if vi.isValid() {
		t.Fatalf("Expected VersionInfo to be invalid")
	}
	vi = VersionInfo{"foo", ""}
	if vi.isValid() {
		t.Fatalf("Expected VersionInfo to be invalid")
	}
}

func TestAppendVersions(t *testing.T) {
	vis := []VersionInfo{
		{"foo", "1.0"},
		{"bar", "0.1"},
		{"pi", "3.1.4"},
	}
	v := AppendVersions("base", vis...)
	expect := "base foo/1.0 bar/0.1 pi/3.1.4"
	if v != expect {
		t.Fatalf("expected %q, got %q", expect, v)
	}
}
