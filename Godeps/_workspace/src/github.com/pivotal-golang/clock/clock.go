package clock

import "time"

type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
	Since(t time.Time) time.Duration

	NewTimer(d time.Duration) Timer
	NewTicker(d time.Duration) Ticker
}

type realClock struct{}

func NewClock() Clock {
	return &realClock{}
}

func (clock *realClock) Now() time.Time {
	return time.Now()
}

func (clock *realClock) Since(t time.Time) time.Duration {
	return time.Now().Sub(t)
}

func (clock *realClock) Sleep(d time.Duration) {
	<-clock.NewTimer(d).C()
}

func (clock *realClock) NewTimer(d time.Duration) Timer {
	return &realTimer{
		t: time.NewTimer(d),
	}
}

func (clock *realClock) NewTicker(d time.Duration) Ticker {
	return &realTicker{
		t: time.NewTicker(d),
	}
}
