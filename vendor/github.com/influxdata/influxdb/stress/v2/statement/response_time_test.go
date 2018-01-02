package statement

import (
	"testing"
)

func TestNewResponseTime(t *testing.T) {
	value := 100000
	rs := NewResponseTime(value)
	if rs.Value != value {
		t.Errorf("expected: %v\ngot: %v\n", value, rs.Value)
	}
}

func newResponseTimes() ResponseTimes {
	return []ResponseTime{
		NewResponseTime(100),
		NewResponseTime(10),
	}
}

func TestResponseTimeLen(t *testing.T) {
	rs := newResponseTimes()
	if rs.Len() != 2 {
		t.Fail()
	}
}

func TestResponseTimeLess(t *testing.T) {
	rs := newResponseTimes()
	less := rs.Less(1, 0)
	if !less {
		t.Fail()
	}
}

func TestResponseTimeSwap(t *testing.T) {
	rs := newResponseTimes()
	rs0 := rs[0]
	rs1 := rs[1]
	rs.Swap(0, 1)
	if rs0 != rs[1] || rs1 != rs[0] {
		t.Fail()
	}
}
