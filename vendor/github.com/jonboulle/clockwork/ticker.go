package clockwork

import (
	"time"
)

// Ticker provides an interface which can be used instead of directly
// using the ticker within the time module. The real-time ticker t
// provides ticks through t.C which becomes now t.Chan() to make
// this channel requirement definable in this interface.
type Ticker interface {
	Chan() <-chan time.Time
	Stop()
}

type realTicker struct{ *time.Ticker }

func (rt *realTicker) Chan() <-chan time.Time {
	return rt.C
}

type fakeTicker struct {
	c      chan time.Time
	stop   chan bool
	clock  FakeClock
	period time.Duration
}

func (ft *fakeTicker) Chan() <-chan time.Time {
	return ft.c
}

func (ft *fakeTicker) Stop() {
	ft.stop <- true
}

// runTickThread initializes a background goroutine to send the tick time to the ticker channel
// after every period. Tick events are discarded if the underlying ticker channel does not have
// enough capacity.
func (ft *fakeTicker) runTickThread() {
	nextTick := ft.clock.Now().Add(ft.period)
	next := ft.clock.After(ft.period)
	go func() {
		for {
			select {
			case <-ft.stop:
				return
			case <-next:
				// We send the time that the tick was supposed to occur at.
				tick := nextTick
				// Before sending the tick, we'll compute the next tick time and star the clock.After call.
				now := ft.clock.Now()
				// First, figure out how many periods there have been between "now" and the time we were
				// supposed to have trigged, then advance over all of those.
				skipTicks := (now.Sub(tick) + ft.period - 1) / ft.period
				nextTick = nextTick.Add(skipTicks * ft.period)
				// Now, keep advancing until we are past now. This should happen at most once.
				for !nextTick.After(now) {
					nextTick = nextTick.Add(ft.period)
				}
				// Figure out how long between now and the next scheduled tick, then wait that long.
				remaining := nextTick.Sub(now)
				next = ft.clock.After(remaining)
				// Finally, we can actually send the tick.
				select {
				case ft.c <- tick:
				default:
				}
			}
		}
	}()
}
