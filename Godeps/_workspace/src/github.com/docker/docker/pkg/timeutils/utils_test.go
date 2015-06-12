package timeutils

import (
	"fmt"
	"testing"
	"time"
)

func TestGetTimestamp(t *testing.T) {
	now := time.Now()
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

		// Durations
		{"1m", fmt.Sprintf("%d", now.Add(-1*time.Minute).Unix())},
		{"1.5h", fmt.Sprintf("%d", now.Add(-90*time.Minute).Unix())},
		{"1h30m", fmt.Sprintf("%d", now.Add(-90*time.Minute).Unix())},

		// String fallback
		{"invalid", "invalid"},
	}

	for _, c := range cases {
		o := GetTimestamp(c.in, now)
		if o != c.expected {
			t.Fatalf("wrong value for '%s'. expected:'%s' got:'%s'", c.in, c.expected, o)
		}
	}
}
