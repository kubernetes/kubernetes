package stressClient

import (
	"crypto/rand"
	"fmt"
	"log"
	"strconv"
	"sync"
)

// ###########################################
// ConcurrencyLimiter and associated methods #
// ###########################################

// ConcurrencyLimiter ensures that no more than a specified
// max number of goroutines are running.
type ConcurrencyLimiter struct {
	inc   chan chan struct{}
	dec   chan struct{}
	max   int
	count int

	sync.Mutex
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

// NewMax resets the max of a ConcurrencyLimiter.
func (c *ConcurrencyLimiter) NewMax(i int) {
	c.Lock()
	defer c.Unlock()
	c.max = i
}

// handleLimits runs in a goroutine to manage the count of
// running goroutines.
func (c *ConcurrencyLimiter) handleLimits() {
	for {
		r := <-c.inc
		c.Lock()
		if c.count >= c.max {
			<-c.dec
			c.count--
		}
		c.Unlock()
		c.count++
		r <- struct{}{}
	}
}

// Utility interger parsing function
func parseInt(s string) int {
	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		log.Fatalf("Error parsing integer:\n  String: %v\n  Error: %v\n", s, err)
	}
	return int(i)
}

// Utility for making random strings of length n
func randStr(n int) string {
	b := make([]byte, n/2)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}
