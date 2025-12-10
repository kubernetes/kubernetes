package backoff

import (
	"context"
	"sync"
	"time"
)

// Ticker holds a channel that delivers `ticks' of a clock at times reported by a BackOff.
//
// Ticks will continue to arrive when the previous operation is still running,
// so operations that take a while to fail could run in quick succession.
type Ticker struct {
	C        <-chan time.Time
	c        chan time.Time
	b        BackOff
	ctx      context.Context
	timer    Timer
	stop     chan struct{}
	stopOnce sync.Once
}

// NewTicker returns a new Ticker containing a channel that will send
// the time at times specified by the BackOff argument. Ticker is
// guaranteed to tick at least once.  The channel is closed when Stop
// method is called or BackOff stops. It is not safe to manipulate the
// provided backoff policy (notably calling NextBackOff or Reset)
// while the ticker is running.
func NewTicker(b BackOff) *Ticker {
	return NewTickerWithTimer(b, &defaultTimer{})
}

// NewTickerWithTimer returns a new Ticker with a custom timer.
// A default timer that uses system timer is used when nil is passed.
func NewTickerWithTimer(b BackOff, timer Timer) *Ticker {
	if timer == nil {
		timer = &defaultTimer{}
	}
	c := make(chan time.Time)
	t := &Ticker{
		C:     c,
		c:     c,
		b:     b,
		ctx:   getContext(b),
		timer: timer,
		stop:  make(chan struct{}),
	}
	t.b.Reset()
	go t.run()
	return t
}

// Stop turns off a ticker. After Stop, no more ticks will be sent.
func (t *Ticker) Stop() {
	t.stopOnce.Do(func() { close(t.stop) })
}

func (t *Ticker) run() {
	c := t.c
	defer close(c)

	// Ticker is guaranteed to tick at least once.
	afterC := t.send(time.Now())

	for {
		if afterC == nil {
			return
		}

		select {
		case tick := <-afterC:
			afterC = t.send(tick)
		case <-t.stop:
			t.c = nil // Prevent future ticks from being sent to the channel.
			return
		case <-t.ctx.Done():
			return
		}
	}
}

func (t *Ticker) send(tick time.Time) <-chan time.Time {
	select {
	case t.c <- tick:
	case <-t.stop:
		return nil
	}

	next := t.b.NextBackOff()
	if next == Stop {
		t.Stop()
		return nil
	}

	t.timer.Start(next)
	return t.timer.C()
}
