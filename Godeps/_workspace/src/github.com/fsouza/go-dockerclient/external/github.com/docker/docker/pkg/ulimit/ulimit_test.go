package ulimit

import "testing"

func TestParseValid(t *testing.T) {
	u1 := &Ulimit{"nofile", 1024, 512}
	if u2, _ := Parse("nofile=512:1024"); *u1 != *u2 {
		t.Fatalf("expected %q, but got %q", u1, u2)
	}
}

func TestParseInvalidLimitType(t *testing.T) {
	if _, err := Parse("notarealtype=1024:1024"); err == nil {
		t.Fatalf("expected error on invalid ulimit type")
	}
}

func TestParseBadFormat(t *testing.T) {
	if _, err := Parse("nofile:1024:1024"); err == nil {
		t.Fatal("expected error on bad syntax")
	}

	if _, err := Parse("nofile"); err == nil {
		t.Fatal("expected error on bad syntax")
	}

	if _, err := Parse("nofile="); err == nil {
		t.Fatal("expected error on bad syntax")
	}
	if _, err := Parse("nofile=:"); err == nil {
		t.Fatal("expected error on bad syntax")
	}
	if _, err := Parse("nofile=:1024"); err == nil {
		t.Fatal("expected error on bad syntax")
	}
}

func TestParseHardLessThanSoft(t *testing.T) {
	if _, err := Parse("nofile:1024:1"); err == nil {
		t.Fatal("expected error on hard limit less than soft limit")
	}
}

func TestParseInvalidValueType(t *testing.T) {
	if _, err := Parse("nofile:asdf"); err == nil {
		t.Fatal("expected error on bad value type")
	}
}

func TestStringOutput(t *testing.T) {
	u := &Ulimit{"nofile", 1024, 512}
	if s := u.String(); s != "nofile=512:1024" {
		t.Fatal("expected String to return nofile=512:1024, but got", s)
	}
}
