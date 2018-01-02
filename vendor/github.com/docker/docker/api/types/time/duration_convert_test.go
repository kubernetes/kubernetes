package time

import (
	"testing"
	"time"
)

func TestDurationToSecondsString(t *testing.T) {
	cases := []struct {
		in       time.Duration
		expected string
	}{
		{0 * time.Second, "0"},
		{1 * time.Second, "1"},
		{1 * time.Minute, "60"},
		{24 * time.Hour, "86400"},
	}

	for _, c := range cases {
		s := DurationToSecondsString(c.in)
		if s != c.expected {
			t.Errorf("wrong value for input `%v`: expected `%s`, got `%s`", c.in, c.expected, s)
			t.Fail()
		}
	}
}
