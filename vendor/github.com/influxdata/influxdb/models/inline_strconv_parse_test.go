package models

import (
	"strconv"
	"testing"
	"testing/quick"
)

func TestParseIntBytesEquivalenceFuzz(t *testing.T) {
	f := func(b []byte, base int, bitSize int) bool {
		exp, expErr := strconv.ParseInt(string(b), base, bitSize)
		got, gotErr := parseIntBytes(b, base, bitSize)

		return exp == got && checkErrs(expErr, gotErr)
	}

	cfg := &quick.Config{
		MaxCount: 10000,
	}

	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestParseIntBytesValid64bitBase10EquivalenceFuzz(t *testing.T) {
	buf := []byte{}
	f := func(n int64) bool {
		buf = strconv.AppendInt(buf[:0], n, 10)

		exp, expErr := strconv.ParseInt(string(buf), 10, 64)
		got, gotErr := parseIntBytes(buf, 10, 64)

		return exp == got && checkErrs(expErr, gotErr)
	}

	cfg := &quick.Config{
		MaxCount: 10000,
	}

	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestParseFloatBytesEquivalenceFuzz(t *testing.T) {
	f := func(b []byte, bitSize int) bool {
		exp, expErr := strconv.ParseFloat(string(b), bitSize)
		got, gotErr := parseFloatBytes(b, bitSize)

		return exp == got && checkErrs(expErr, gotErr)
	}

	cfg := &quick.Config{
		MaxCount: 10000,
	}

	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestParseFloatBytesValid64bitEquivalenceFuzz(t *testing.T) {
	buf := []byte{}
	f := func(n float64) bool {
		buf = strconv.AppendFloat(buf[:0], n, 'f', -1, 64)

		exp, expErr := strconv.ParseFloat(string(buf), 64)
		got, gotErr := parseFloatBytes(buf, 64)

		return exp == got && checkErrs(expErr, gotErr)
	}

	cfg := &quick.Config{
		MaxCount: 10000,
	}

	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestParseBoolBytesEquivalence(t *testing.T) {
	var buf []byte
	for _, s := range []string{"1", "t", "T", "TRUE", "true", "True", "0", "f", "F", "FALSE", "false", "False", "fail", "TrUe", "FAlSE", "numbers", ""} {
		buf = append(buf[:0], s...)

		exp, expErr := strconv.ParseBool(s)
		got, gotErr := parseBoolBytes(buf)

		if got != exp || !checkErrs(expErr, gotErr) {
			t.Errorf("Failed to parse boolean value %q correctly: wanted (%t, %v), got (%t, %v)", s, exp, expErr, got, gotErr)
		}
	}
}

func checkErrs(a, b error) bool {
	if (a == nil) != (b == nil) {
		return false
	}

	return a == nil || a.Error() == b.Error()
}
