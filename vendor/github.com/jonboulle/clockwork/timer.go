package clockwork

import "time"

// Timer provides an interface which can be used instead of directly using
// [time.Timer]. The real-time timer t provides events through t.C which becomes
// t.Chan() to make this channel requirement definable in this interface.
type Timer interface {
	Chan() <-chan time.Time
	Reset(d time.Duration) bool
	Stop() bool
}

type realTimer struct{ *time.Timer }

func (r realTimer) Chan() <-chan time.Time {
	return r.C
}

type fakeTimer struct {
	// The channel associated with the firer, used to send expiration times.
	c chan time.Time

	// The time when the firer expires. Only meaningful if the firer is currently
	// one of a FakeClock's waiters.
	exp time.Time

	// reset and stop provide the implementation of the respective exported
	// functions.
	reset func(d time.Duration) bool
	stop  func() bool

	// If present when the timer fires, the timer calls afterFunc in its own
	// goroutine rather than sending the time on Chan().
	afterFunc func()
}

func newFakeTimer(fc *FakeClock, afterfunc func()) *fakeTimer {
	var ft *fakeTimer
	ft = &fakeTimer{
		c: make(chan time.Time, 1),
		reset: func(d time.Duration) bool {
			fc.l.Lock()
			defer fc.l.Unlock()
			// fc.l must be held across the calls to stopExpirer & setExpirer.
			stopped := fc.stopExpirer(ft)
			fc.setExpirer(ft, d)
			return stopped
		},
		stop: func() bool { return fc.stop(ft) },

		afterFunc: afterfunc,
	}
	return ft
}

func (f *fakeTimer) Chan() <-chan time.Time { return f.c }

func (f *fakeTimer) Reset(d time.Duration) bool { return f.reset(d) }

func (f *fakeTimer) Stop() bool { return f.stop() }

func (f *fakeTimer) expire(now time.Time) *time.Duration {
	if f.afterFunc != nil {
		go f.afterFunc()
		return nil
	}

	// Never block on expiration.
	select {
	case f.c <- now:
	default:
	}
	return nil
}

func (f *fakeTimer) expiration() time.Time { return f.exp }

func (f *fakeTimer) setExpiration(t time.Time) { f.exp = t }
