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
	// The channel associated with the firer, used to send expiration times.
	c chan time.Time

	// The time when the ticker expires. Only meaningful if the ticker is currently
	// one of a FakeClock's waiters.
	exp time.Time

	// reset and stop provide the implementation of the respective exported
	// functions.
	reset func(d time.Duration)
	stop  func()

	// The duration of the ticker.
	d time.Duration
}

func newFakeTicker(fc *FakeClock, d time.Duration) *fakeTicker {
	var ft *fakeTicker
	ft = &fakeTicker{
		c: make(chan time.Time, 1),
		d: d,
		reset: func(d time.Duration) {
			fc.l.Lock()
			defer fc.l.Unlock()
			ft.d = d
			fc.setExpirer(ft, d)
		},
		stop: func() { fc.stop(ft) },
	}
	return ft
}

func (f *fakeTicker) Chan() <-chan time.Time { return f.c }

func (f *fakeTicker) Reset(d time.Duration) { f.reset(d) }

func (f *fakeTicker) Stop() { f.stop() }

func (f *fakeTicker) expire(now time.Time) *time.Duration {
	// Never block on expiration.
	select {
	case f.c <- now:
	default:
	}
	return &f.d
}

func (f *fakeTicker) expiration() time.Time { return f.exp }

func (f *fakeTicker) setExpiration(t time.Time) { f.exp = t }
