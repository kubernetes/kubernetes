package mergo

import (
	"testing"
	"time"
)

type testStruct struct {
	time.Duration
}

func TestIssue50Merge(t *testing.T) {
	to := testStruct{}
	from := testStruct{}
	if err := Merge(&to, from); err != nil {
		t.Fail()
	}
}
