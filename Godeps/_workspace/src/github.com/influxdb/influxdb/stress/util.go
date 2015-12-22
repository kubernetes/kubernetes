package stress

import (
	"time"
)

// Timer is struct that can be used to track elaspsed time
type Timer struct {
	start time.Time
	end   time.Time
}

// Start returns a Timers start field
func (t *Timer) Start() time.Time {
	return t.start
}

// End returns a Timers end field
func (t *Timer) End() time.Time {
	return t.end
}

// StartTimer sets a timers `start` field to the current time
func (t *Timer) StartTimer() {
	t.start = time.Now()
}

// StopTimer sets a timers `end` field to the current time
func (t *Timer) StopTimer() {
	t.end = time.Now()
}

// Elapsed returns the total elapsed time between the `start`
// and `end` fields on a timer.
func (t *Timer) Elapsed() time.Duration {
	return t.end.Sub(t.start)
}

// NewTimer returns a pointer to a `Timer` struct where the
// timers `start` field has been set to `time.Now()`
func NewTimer() *Timer {
	t := &Timer{}
	t.StartTimer()
	return t
}

// ResponseTime is a struct that contains `Value`
// `Time` pairing.
type ResponseTime struct {
	Value int
	Time  time.Time
}

// NewResponseTime returns a new response time
// with value `v` and time `time.Now()`.
func NewResponseTime(v int) ResponseTime {
	r := ResponseTime{Value: v, Time: time.Now()}
	return r
}

// ResponseTimes is a slice of response times
type ResponseTimes []ResponseTime

// Implements the `Len` method for the
// sort.Interface type
func (rs ResponseTimes) Len() int {
	return len(rs)
}

// Implements the `Less` method for the
// sort.Interface type
func (rs ResponseTimes) Less(i, j int) bool {
	return rs[i].Value < rs[j].Value
}

// Implements the `Swap` method for the
// sort.Interface type
func (rs ResponseTimes) Swap(i, j int) {
	rs[i], rs[j] = rs[j], rs[i]
}

//////////////////////////////////

// ConcurrencyLimiter is a go routine safe struct that can be used to
// ensure that no more than a specifid max number of goroutines are
// executing.
type ConcurrencyLimiter struct {
	inc   chan chan struct{}
	dec   chan struct{}
	max   int
	count int
}

// NewConcurrencyLimiter returns a configured limiter that will
// ensure that calls to Increment will block if the max is hit.
func NewConcurrencyLimiter(max int) *ConcurrencyLimiter {
	c := &ConcurrencyLimiter{
		inc: make(chan chan struct{}),
		dec: make(chan struct{}, max),
		max: max,
	}
	go c.handleLimits()
	return c
}

// Increment will increase the count of running goroutines by 1.
// if the number is currently at the max, the call to Increment
// will block until another goroutine decrements.
func (c *ConcurrencyLimiter) Increment() {
	r := make(chan struct{})
	c.inc <- r
	<-r
}

// Decrement will reduce the count of running goroutines by 1
func (c *ConcurrencyLimiter) Decrement() {
	c.dec <- struct{}{}
}

// handleLimits runs in a goroutine to manage the count of
// running goroutines.
func (c *ConcurrencyLimiter) handleLimits() {
	for {
		r := <-c.inc
		if c.count >= c.max {
			<-c.dec
			c.count--
		}
		c.count++
		r <- struct{}{}
	}
}
