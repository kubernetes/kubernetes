package probing

import (
	"sync"
	"time"
)

var (
	// weight factor
	α = 0.125
)

type Status interface {
	Total() int64
	Loss() int64
	Health() bool
	Err() error
	// Estimated smoothed round trip time
	SRTT() time.Duration
	// Estimated clock difference
	ClockDiff() time.Duration
	StopNotify() <-chan struct{}
}

type status struct {
	mu        sync.Mutex
	srtt      time.Duration
	total     int64
	loss      int64
	health    bool
	err       error
	clockdiff time.Duration
	stopC     chan struct{}
}

// SRTT = (1-α) * SRTT + α * RTT
func (s *status) SRTT() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.srtt
}

func (s *status) Total() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}

func (s *status) Loss() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loss
}

func (s *status) Health() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.health
}

func (s *status) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *status) ClockDiff() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.clockdiff
}

func (s *status) StopNotify() <-chan struct{} {
	return s.stopC
}

func (s *status) record(rtt time.Duration, when time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.total += 1
	s.health = true
	s.srtt = time.Duration((1-α)*float64(s.srtt) + α*float64(rtt))
	s.clockdiff = time.Now().Sub(when) - s.srtt/2
	s.err = nil
}

func (s *status) recordFailure(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.total++
	s.health = false
	s.loss += 1
	s.err = err
}

func (s *status) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.srtt = 0
	s.total = 0
	s.loss = 0
	s.health = false
	s.clockdiff = 0
	s.err = nil
}
