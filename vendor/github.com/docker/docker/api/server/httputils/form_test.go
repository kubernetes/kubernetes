package httputils

import (
	"net/http"
	"net/url"
	"testing"
)

func TestBoolValue(t *testing.T) {
	cases := map[string]bool{
		"":      false,
		"0":     false,
		"no":    false,
		"false": false,
		"none":  false,
		"1":     true,
		"yes":   true,
		"true":  true,
		"one":   true,
		"100":   true,
	}

	for c, e := range cases {
		v := url.Values{}
		v.Set("test", c)
		r, _ := http.NewRequest("POST", "", nil)
		r.Form = v

		a := BoolValue(r, "test")
		if a != e {
			t.Fatalf("Value: %s, expected: %v, actual: %v", c, e, a)
		}
	}
}

func TestBoolValueOrDefault(t *testing.T) {
	r, _ := http.NewRequest("GET", "", nil)
	if !BoolValueOrDefault(r, "queryparam", true) {
		t.Fatal("Expected to get true default value, got false")
	}

	v := url.Values{}
	v.Set("param", "")
	r, _ = http.NewRequest("GET", "", nil)
	r.Form = v
	if BoolValueOrDefault(r, "param", true) {
		t.Fatal("Expected not to get true")
	}
}

func TestInt64ValueOrZero(t *testing.T) {
	cases := map[string]int64{
		"":     0,
		"asdf": 0,
		"0":    0,
		"1":    1,
	}

	for c, e := range cases {
		v := url.Values{}
		v.Set("test", c)
		r, _ := http.NewRequest("POST", "", nil)
		r.Form = v

		a := Int64ValueOrZero(r, "test")
		if a != e {
			t.Fatalf("Value: %s, expected: %v, actual: %v", c, e, a)
		}
	}
}

func TestInt64ValueOrDefault(t *testing.T) {
	cases := map[string]int64{
		"":   -1,
		"-1": -1,
		"42": 42,
	}

	for c, e := range cases {
		v := url.Values{}
		v.Set("test", c)
		r, _ := http.NewRequest("POST", "", nil)
		r.Form = v

		a, err := Int64ValueOrDefault(r, "test", -1)
		if a != e {
			t.Fatalf("Value: %s, expected: %v, actual: %v", c, e, a)
		}
		if err != nil {
			t.Fatalf("Error should be nil, but received: %s", err)
		}
	}
}

func TestInt64ValueOrDefaultWithError(t *testing.T) {
	v := url.Values{}
	v.Set("test", "invalid")
	r, _ := http.NewRequest("POST", "", nil)
	r.Form = v

	_, err := Int64ValueOrDefault(r, "test", -1)
	if err == nil {
		t.Fatal("Expected an error.")
	}
}
