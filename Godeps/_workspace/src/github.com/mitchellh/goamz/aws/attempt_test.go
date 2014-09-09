package aws_test

import (
	"github.com/mitchellh/goamz/aws"
	. "github.com/motain/gocheck"
	"time"
)

func (S) TestAttemptTiming(c *C) {
	testAttempt := aws.AttemptStrategy{
		Total: 0.25e9,
		Delay: 0.1e9,
	}
	want := []time.Duration{0, 0.1e9, 0.2e9, 0.2e9}
	got := make([]time.Duration, 0, len(want)) // avoid allocation when testing timing
	t0 := time.Now()
	for a := testAttempt.Start(); a.Next(); {
		got = append(got, time.Now().Sub(t0))
	}
	got = append(got, time.Now().Sub(t0))
	c.Assert(got, HasLen, len(want))
	const margin = 0.01e9
	for i, got := range want {
		lo := want[i] - margin
		hi := want[i] + margin
		if got < lo || got > hi {
			c.Errorf("attempt %d want %g got %g", i, want[i].Seconds(), got.Seconds())
		}
	}
}

func (S) TestAttemptNextHasNext(c *C) {
	a := aws.AttemptStrategy{}.Start()
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.Next(), Equals, false)

	a = aws.AttemptStrategy{}.Start()
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.HasNext(), Equals, false)
	c.Assert(a.Next(), Equals, false)

	a = aws.AttemptStrategy{Total: 2e8}.Start()
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.HasNext(), Equals, true)
	time.Sleep(2e8)
	c.Assert(a.HasNext(), Equals, true)
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.Next(), Equals, false)

	a = aws.AttemptStrategy{Total: 1e8, Min: 2}.Start()
	time.Sleep(1e8)
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.HasNext(), Equals, true)
	c.Assert(a.Next(), Equals, true)
	c.Assert(a.HasNext(), Equals, false)
	c.Assert(a.Next(), Equals, false)
}
