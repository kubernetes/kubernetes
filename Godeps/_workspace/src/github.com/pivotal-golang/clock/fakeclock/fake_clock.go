package fakeclock

import (
	"sync"
	"time"

	"github.com/pivotal-golang/clock"
)

type timeWatcher interface {
	timeUpdated(time.Time)
}

type FakeClock struct {
	sync.Mutex
	now time.Time

	watchers map[timeWatcher]struct{}
}

func NewFakeClock(now time.Time) *FakeClock {
	return &FakeClock{
		now:      now,
		watchers: make(map[timeWatcher]struct{}),
	}
}
func (clock *FakeClock) Since(t time.Time) time.Duration {
	return clock.Now().Sub(t)
}

func (clock *FakeClock) Now() time.Time {
	clock.Mutex.Lock()
	defer clock.Mutex.Unlock()

	return clock.now
}

func (clock *FakeClock) Increment(duration time.Duration) {
	clock.Mutex.Lock()
	now := clock.now.Add(duration)
	clock.now = now

	watchers := make([]timeWatcher, 0, len(clock.watchers))
	for w, _ := range clock.watchers {
		watchers = append(watchers, w)
	}
	clock.Mutex.Unlock()

	for _, w := range watchers {
		w.timeUpdated(now)
	}
}

func (clock *FakeClock) IncrementBySeconds(seconds uint64) {
	clock.Increment(time.Duration(seconds) * time.Second)
}

func (clock *FakeClock) NewTimer(d time.Duration) clock.Timer {
	timer := NewFakeTimer(clock, d)
	clock.addTimeWatcher(timer)

	return timer
}

func (clock *FakeClock) Sleep(d time.Duration) {
	<-clock.NewTimer(d).C()
}

func (clock *FakeClock) NewTicker(d time.Duration) clock.Ticker {
	return NewFakeTicker(clock, d)
}

func (clock *FakeClock) WatcherCount() int {
	clock.Mutex.Lock()
	defer clock.Mutex.Unlock()

	return len(clock.watchers)
}

func (clock *FakeClock) addTimeWatcher(tw timeWatcher) {
	clock.Mutex.Lock()
	clock.watchers[tw] = struct{}{}
	clock.Mutex.Unlock()

	tw.timeUpdated(clock.Now())
}

func (clock *FakeClock) removeTimeWatcher(tw timeWatcher) {
	clock.Mutex.Lock()
	delete(clock.watchers, tw)
	clock.Mutex.Unlock()
}
