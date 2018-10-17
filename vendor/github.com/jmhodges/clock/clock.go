// Package clock provides an abstraction for system time that enables
// testing of time-sensitive code.
//
// Where you'd use time.Now, instead use clk.Now where clk is an
// instance of Clock.
//
// When running your code in production, pass it a Clock given by
// Default() and when you're running it in your tests, pass it an
// instance of Clock from NewFake().
//
// When you do that, you can use FakeClock's Add and Set methods to
// control how time behaves in your code making them more reliable
// while also expanding the space of problems you can test.
//
// This code intentionally does not attempt to provide an abstraction
// over time.Ticker and time.Timer because Go does not have the
// runtime or API hooks available to do reliably. See
// https://github.com/golang/go/issues/8869
//
// Be sure to test Time equality with time.Time#Equal, not ==.
package clock

import (
	"sort"
	"sync"
	"time"
)

// Some in-use reflection-heavy systems, like facebookgo/inject, fail when given
// a value type like sysClock{}. Since it's hidden by an interface, this has
// surprised users. We fixed that by making systemClock a &sysClock.
var systemClock Clock = &sysClock{}

// New returns a Clock that matches the actual system time.
func New() Clock {
	// This is a method instead of a public var to prevent folks from
	// "making things work" by writing to the var instead of passing
	// in a Clock.
	return systemClock
}

// Deprecated: Default is just an alias for New but less memorable.
func Default() Clock {
	return systemClock
}

// Clock is an abstraction over system time. New instances of it can
// be made with Default and NewFake.
type Clock interface {
	// Now returns the Clock's current view of the time. Mutating the
	// returned Time will not mutate the clock's time.
	Now() time.Time

	// Sleep causes the current goroutine to sleep for the given duration.
	Sleep(time.Duration)

	// After returns a channel that fires after the given duration.
	After(time.Duration) <-chan time.Time

	// Since is a short hand for Now().Sub(t).
	Since(time.Time) time.Duration

	// NewTimer makes a Timer based on this clock's time. Using Timers and
	// negative durations in the Clock or Timer API is undefined behavior and
	// may be changed.
	NewTimer(time.Duration) *Timer
}

type sysClock struct{}

func (s *sysClock) Now() time.Time {
	return time.Now()
}

func (s *sysClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

func (s *sysClock) After(d time.Duration) <-chan time.Time {
	return time.After(d)
}

func (s *sysClock) Since(t time.Time) time.Duration {
	return time.Since(t)
}

func (s *sysClock) NewTimer(d time.Duration) *Timer {
	tt := time.NewTimer(d)
	return &Timer{C: tt.C, timer: tt}
}

// NewFake returns a FakeClock to be used in tests that need to
// manipulate time. Its initial value is always the unix epoch in the
// UTC timezone. The FakeClock returned is thread-safe.
func NewFake() FakeClock {
	// We're explicit about this time construction to avoid early user
	// questions about why the time object doesn't have a Location by
	// default.
	return &fake{t: time.Unix(0, 0).UTC()}
}

// FakeClock is a Clock with additional controls. The return value of
// Now return can be modified with Add. Use NewFake to get a
// thread-safe FakeClock implementation.
type FakeClock interface {
	Clock
	// Adjust the time that will be returned by Now.
	Add(d time.Duration)

	// Set the Clock's time to exactly the time given.
	Set(t time.Time)
}

// To prevent mistakes with the API, we hide this behind NewFake. It's
// easy forget to create a pointer to a fake since time.Time (and
// sync.Mutex) are also simple values. The code will appear to work
// but the clock's time will never be adjusted.
type fake struct {
	sync.RWMutex
	t     time.Time
	sends sortedSends
}

func (f *fake) Now() time.Time {
	f.RLock()
	defer f.RUnlock()
	return f.t
}

func (f *fake) Sleep(d time.Duration) {
	if d < 0 {
		// time.Sleep just returns immediately. Do the same.
		return
	}
	f.Add(d)
}

func (f *fake) After(d time.Duration) <-chan time.Time {
	return f.NewTimer(d).C
}

func (f *fake) Since(t time.Time) time.Duration {
	return f.Now().Sub(t)
}

func (f *fake) NewTimer(d time.Duration) *Timer {
	f.Lock()
	defer f.Unlock()
	ch := make(chan time.Time, 1)
	tt := f.t.Add(d)
	ft := &fakeTimer{c: ch, clk: f, active: true}
	t := &Timer{
		C:         ch,
		fakeTimer: ft,
	}
	s := f.addSend(tt, ft)
	ft.sends = []*send{s}
	return t
}

func (f *fake) Add(d time.Duration) {
	f.Lock()
	defer f.Unlock()
	f.t = f.t.Add(d)
	f.sendTimes()
}

func (f *fake) Set(t time.Time) {
	f.Lock()
	defer f.Unlock()
	f.t = t
	f.sendTimes()
}

// Only to be called while the fake's lock is held
func (f *fake) sendTimes() {
	newSends := make(sortedSends, 0)
	for _, s := range f.sends {
		if !s.active || !s.ft.active {
			continue
		}
		if s.target.Equal(f.t) || s.target.Before(f.t) {
			s.ft.active = false
			s.active = false
			// The select is to drop second sends from resets without a user
			// receiving from ft.c.
			select {
			case s.ft.c <- s.target:
			default:
			}
		}
		if s.active {
			newSends = append(newSends, s)
		}
	}
	f.sends = newSends
}

// Only to be called while the fake's lock is held
func (f *fake) addSend(target time.Time, ft *fakeTimer) *send {
	s := &send{target: target, ft: ft, active: true}
	f.sends = append(f.sends, s)
	// This will be a small enough slice to be fast. Can be replaced with a more
	// complicated container if someone is making many timers.
	sort.Sort(f.sends)
	return s
}

// send is a struct that represents a scheduled send of a time.Time to its
// fakeTimer's channel. They are actually sent when the relevant fake's time
// goes equal or past their target time, as long as the relevant fakeTimer has
// not been Reset or Stop'ed. When a Timer is Reset, the old sends are
// deactivated and will be removed from the clocks list on the next attempt to
// send.
type send struct {
	target time.Time
	active bool
	ft     *fakeTimer
}

type sortedSends []*send

func (s sortedSends) Len() int {
	return len(s)
}

func (s sortedSends) Less(i, j int) bool {
	return s[i].target.Before(s[j].target)
}

func (s sortedSends) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
