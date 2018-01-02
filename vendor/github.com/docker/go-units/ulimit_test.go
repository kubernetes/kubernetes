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

func TestParseUlimitTooManyValueArgs(t *testing.T) {
	if _, err := ParseUlimit("nofile=1024:1:50"); err == nil {
		t.Fatalf("expected error on more than two value arguments")
	}
}

func TestUlimitStringOutput(t *testing.T) {
	u := &Ulimit{"nofile", 1024, 512}
	if s := u.String(); s != "nofile=512:1024" {
		t.Fatal("expected String to return nofile=512:1024, but got", s)
	}
}

func TestGetRlimit(t *testing.T) {
	tt := []struct {
		ulimit Ulimit
		rlimit Rlimit
	}{
		{Ulimit{"core", 10, 12}, Rlimit{rlimitCore, 10, 12}},
		{Ulimit{"cpu", 1, 10}, Rlimit{rlimitCPU, 1, 10}},
		{Ulimit{"data", 5, 0}, Rlimit{rlimitData, 5, 0}},
		{Ulimit{"fsize", 2, 2}, Rlimit{rlimitFsize, 2, 2}},
		{Ulimit{"locks", 0, 0}, Rlimit{rlimitLocks, 0, 0}},
		{Ulimit{"memlock", 10, 10}, Rlimit{rlimitMemlock, 10, 10}},
		{Ulimit{"msgqueue", 9, 1}, Rlimit{rlimitMsgqueue, 9, 1}},
		{Ulimit{"nice", 9, 9}, Rlimit{rlimitNice, 9, 9}},
		{Ulimit{"nofile", 4, 100}, Rlimit{rlimitNofile, 4, 100}},
		{Ulimit{"nproc", 5, 5}, Rlimit{rlimitNproc, 5, 5}},
		{Ulimit{"rss", 0, 5}, Rlimit{rlimitRss, 0, 5}},
		{Ulimit{"rtprio", 100, 65}, Rlimit{rlimitRtprio, 100, 65}},
		{Ulimit{"rttime", 55, 102}, Rlimit{rlimitRttime, 55, 102}},
		{Ulimit{"sigpending", 14, 20}, Rlimit{rlimitSigpending, 14, 20}},
		{Ulimit{"stack", 1, 1}, Rlimit{rlimitStack, 1, 1}},
	}

	for _, te := range tt {
		res, err := te.ulimit.GetRlimit()
		if err != nil {
			t.Errorf("expected not to fail: %s", err)
		}
		if res.Type != te.rlimit.Type {
			t.Errorf("expected Type to be %d but got %d",
				te.rlimit.Type, res.Type)
		}
		if res.Soft != te.rlimit.Soft {
			t.Errorf("expected Soft to be %d but got %d",
				te.rlimit.Soft, res.Soft)
		}
		if res.Hard != te.rlimit.Hard {
			t.Errorf("expected Hard to be %d but got %d",
				te.rlimit.Hard, res.Hard)
		}

	}
}

func TestGetRlimitBadUlimitName(t *testing.T) {
	name := "bla"
	uLimit := Ulimit{name, 0, 0}
	if _, err := uLimit.GetRlimit(); err == nil {
		t.Error("expected error on bad Ulimit name")
	}
}
