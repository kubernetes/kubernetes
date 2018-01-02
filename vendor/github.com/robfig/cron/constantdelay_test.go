package cron

import (
	"testing"
	"time"
)

func TestConstantDelayNext(t *testing.T) {
	tests := []struct {
		time     string
		delay    time.Duration
		expected string
	}{
		// Simple cases
		{"Mon Jul 9 14:45 2012", 15*time.Minute + 50*time.Nanosecond, "Mon Jul 9 15:00 2012"},
		{"Mon Jul 9 14:59 2012", 15 * time.Minute, "Mon Jul 9 15:14 2012"},
		{"Mon Jul 9 14:59:59 2012", 15 * time.Minute, "Mon Jul 9 15:14:59 2012"},

		// Wrap around hours
		{"Mon Jul 9 15:45 2012", 35 * time.Minute, "Mon Jul 9 16:20 2012"},

		// Wrap around days
		{"Mon Jul 9 23:46 2012", 14 * time.Minute, "Tue Jul 10 00:00 2012"},
		{"Mon Jul 9 23:45 2012", 35 * time.Minute, "Tue Jul 10 00:20 2012"},
		{"Mon Jul 9 23:35:51 2012", 44*time.Minute + 24*time.Second, "Tue Jul 10 00:20:15 2012"},
		{"Mon Jul 9 23:35:51 2012", 25*time.Hour + 44*time.Minute + 24*time.Second, "Thu Jul 11 01:20:15 2012"},

		// Wrap around months
		{"Mon Jul 9 23:35 2012", 91*24*time.Hour + 25*time.Minute, "Thu Oct 9 00:00 2012"},

		// Wrap around minute, hour, day, month, and year
		{"Mon Dec 31 23:59:45 2012", 15 * time.Second, "Tue Jan 1 00:00:00 2013"},

		// Round to nearest second on the delay
		{"Mon Jul 9 14:45 2012", 15*time.Minute + 50*time.Nanosecond, "Mon Jul 9 15:00 2012"},

		// Round up to 1 second if the duration is less.
		{"Mon Jul 9 14:45:00 2012", 15 * time.Millisecond, "Mon Jul 9 14:45:01 2012"},

		// Round to nearest second when calculating the next time.
		{"Mon Jul 9 14:45:00.005 2012", 15 * time.Minute, "Mon Jul 9 15:00 2012"},

		// Round to nearest second for both.
		{"Mon Jul 9 14:45:00.005 2012", 15*time.Minute + 50*time.Nanosecond, "Mon Jul 9 15:00 2012"},
	}

	for _, c := range tests {
		actual := Every(c.delay).Next(getTime(c.time))
		expected := getTime(c.expected)
		if actual != expected {
			t.Errorf("%s, \"%s\": (expected) %v != %v (actual)", c.time, c.delay, expected, actual)
		}
	}
}
