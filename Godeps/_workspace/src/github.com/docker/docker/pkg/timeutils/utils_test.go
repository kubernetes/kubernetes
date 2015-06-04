package timeutils

import (
	"testing"
)

func TestGetTimestamp(t *testing.T) {
	cases := []struct{ in, expected string }{
		{"0", "-62167305600"}, // 0 gets parsed year 0

		// Partial RFC3339 strings get parsed with second precision
		{"2006-01-02T15:04:05.999999999+07:00", "1136189045"},
		{"2006-01-02T15:04:05.999999999Z", "1136214245"},
		{"2006-01-02T15:04:05.999999999", "1136214245"},
		{"2006-01-02T15:04:05", "1136214245"},
		{"2006-01-02T15:04", "1136214240"},
		{"2006-01-02T15", "1136214000"},
		{"2006-01-02T", "1136160000"},
		{"2006-01-02", "1136160000"},
		{"2006", "1136073600"},
		{"2015-05-13T20:39:09Z", "1431549549"},

		// unix timestamps returned as is
		{"1136073600", "1136073600"},

		// String fallback
		{"invalid", "invalid"},
	}

	for _, c := range cases {
		o := GetTimestamp(c.in)
		if o != c.expected {
			t.Fatalf("wrong value for '%s'. expected:'%s' got:'%s'", c.in, c.expected, o)
		}
	}
}
