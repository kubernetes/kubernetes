package units

import (
	"fmt"
	"strconv"
	"testing"
)

func ExampleParseUlimit() {
	fmt.Println(ParseUlimit("nofile=512:1024"))
	fmt.Println(ParseUlimit("nofile=1024"))
	fmt.Println(ParseUlimit("cpu=2:4"))
	fmt.Println(ParseUlimit("cpu=6"))
}

func TestParseUlimitValid(t *testing.T) {
	u1 := &Ulimit{"nofile", 1024, 512}
	if u2, _ := ParseUlimit("nofile=512:1024"); *u1 != *u2 {
		t.Fatalf("expected %q, but got %q", u1, u2)
	}
}

func TestParseUlimitInvalidLimitType(t *testing.T) {
	if _, err := ParseUlimit("notarealtype=1024:1024"); err == nil {
		t.Fatalf("expected error on invalid ulimit type")
	}
}

func TestParseUlimitBadFormat(t *testing.T) {
	if _, err := ParseUlimit("nofile:1024:1024"); err == nil {
		t.Fatal("expected error on bad syntax")
	}

	if _, err := ParseUlimit("nofile"); err == nil {
		t.Fatal("expected error on bad syntax")
	}

	if _, err := ParseUlimit("nofile="); err == nil {
		t.Fatal("expected error on bad syntax")
	}
	if _, err := ParseUlimit("nofile=:"); err == nil {
		t.Fatal("expected error on bad syntax")
	}
	if _, err := ParseUlimit("nofile=:1024"); err == nil {
		t.Fatal("expected error on bad syntax")
	}
}

func TestParseUlimitHardLessThanSoft(t *testing.T) {
	if _, err := ParseUlimit("nofile=1024:1"); err == nil {
		t.Fatal("expected error on hard limit less than soft limit")
	}
}

func TestParseUlimitInvalidValueType(t *testing.T) {
	if _, err := ParseUlimit("nofile=asdf"); err == nil {
		t.Fatal("expected error on bad value type, but got no error")
	} else if _, ok := err.(*strconv.NumError); !ok {
		t.Fatalf("expected error on bad value type, but got `%s`", err)
	}

	if _, err := ParseUlimit("nofile=1024:asdf"); err == nil {
		t.Fatal("expected error on bad value type, but got no error")
	} else if _, ok := err.(*strconv.NumError); !ok {
		t.Fatalf("expected error on bad value type, but got `%s`", err)
	}
}

func TestUlimitStringOutput(t *testing.T) {
	u := &Ulimit{"nofile", 1024, 512}
	if s := u.String(); s != "nofile=512:1024" {
		t.Fatal("expected String to return nofile=512:1024, but got", s)
	}
}
