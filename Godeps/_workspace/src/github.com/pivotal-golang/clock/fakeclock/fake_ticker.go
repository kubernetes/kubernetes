package fakeclock

import (
	"sync"
	"time"

	"github.com/pivotal-golang/clock"
)

type fakeTicker struct {
	clock clock.Clock

	mutex    sync.Mutex
	duration time.Duration
	channel  chan time.Time

	timer clock.Timer
}

func NewFakeTicker(clock clock.Clock, d time.Duration) clock.Ticker {
	channel := make(chan time.Time)
	timer := clock.NewTimer(d)

	go func() {
		for {
			time := <-timer.C()
			timer.Reset(d)
			channel <- time
		}
	}()

	return &fakeTicker{
		clock:    clock,
		duration: d,
		channel:  channel,
		timer:    timer,
	}
}

func (ft *fakeTicker) C() <-chan time.Time {
	ft.mutex.Lock()
	defer ft.mutex.Unlock()
	return ft.channel
}

func (ft *fakeTicker) Stop() {
	ft.mutex.Lock()
	ft.timer.Stop()
	ft.mutex.Unlock()
}
