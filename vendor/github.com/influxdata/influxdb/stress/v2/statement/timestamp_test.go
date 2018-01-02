package statement

import (
	"testing"
	"time"
)

func TestTimestampTime(t *testing.T) {
	tstp := newTestTimestamp()
	function := tstp.Time("2016-01-01", 100, "s")
	expected := int64(1451606400)
	got := function()
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	function = tstp.Time("now", 100, "ns")
	expected = time.Now().UnixNano()
	got = function()
	if expected < got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func newTestTimestamp() *Timestamp {
	duration, _ := time.ParseDuration("10s")
	return &Timestamp{
		Count:    5001,
		Duration: duration,
		Jitter:   false,
	}
}
