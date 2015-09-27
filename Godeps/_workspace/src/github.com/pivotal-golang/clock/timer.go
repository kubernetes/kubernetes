package clock

import "time"

type Timer interface {
	C() <-chan time.Time
	Reset(d time.Duration) bool
	Stop() bool
}

type realTimer struct {
	t *time.Timer
}

func (t *realTimer) C() <-chan time.Time {
	return t.t.C
}

func (t *realTimer) Reset(d time.Duration) bool {
	return t.t.Reset(d)
}

func (t *realTimer) Stop() bool {
	return t.t.Stop()
}
