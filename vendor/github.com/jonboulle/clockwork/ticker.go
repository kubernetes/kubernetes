package clockwork

import "time"

// Ticker provides an interface which can be used instead of directly using
// [time.Ticker]. The real-time ticker t provides ticks through t.C which
// becomes t.Chan() to make this channel requirement definable in this
// interface.
type Ticker interface {
	Chan() <-chan time.Time
	Reset(d time.Duration)
	Stop()
}

type realTicker struct{ *time.Ticker }

func (r realTicker) Chan() <-chan time.Time {
	return r.C
}

type fakeTicker struct {
	firer

	// reset and stop provide the implementation of the respective exported
	// functions.
	reset func(d time.Duration)
	stop  func()

	// The duration of the ticker.
	d time.Duration
}

func (f *fakeTicker) Reset(d time.Duration) {
	f.reset(d)
}

func (f *fakeTicker) Stop() {
	f.stop()
}

func (f *fakeTicker) expire(now time.Time) *time.Duration {
	// Never block on expiration.
	select {
	case f.c <- now:
	default:
	}
	return &f.d
}
