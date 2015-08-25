//
// Written by Maxim Khitrov (November 2012)
//

// Package flowrate provides the tools for monitoring and limiting the flow rate
// of an arbitrary data stream.
package flowrate

import (
	"math"
	"sync"
	"time"
)

// Monitor monitors and limits the transfer rate of a data stream.
type Monitor struct {
	mu      sync.Mutex    // Mutex guarding access to all internal fields
	active  bool          // Flag indicating an active transfer
	start   time.Duration // Transfer start time (clock() value)
	bytes   int64         // Total number of bytes transferred
	samples int64         // Total number of samples taken

	rSample float64 // Most recent transfer rate sample (bytes per second)
	rEMA    float64 // Exponential moving average of rSample
	rPeak   float64 // Peak transfer rate (max of all rSamples)
	rWindow float64 // rEMA window (seconds)

	sBytes int64         // Number of bytes transferred since sLast
	sLast  time.Duration // Most recent sample time (stop time when inactive)
	sRate  time.Duration // Sampling rate

	tBytes int64         // Number of bytes expected in the current transfer
	tLast  time.Duration // Time of the most recent transfer of at least 1 byte
}

// New creates a new flow control monitor. Instantaneous transfer rate is
// measured and updated for each sampleRate interval. windowSize determines the
// weight of each sample in the exponential moving average (EMA) calculation.
// The exact formulas are:
//
// 	sampleTime = currentTime - prevSampleTime
// 	sampleRate = byteCount / sampleTime
// 	weight     = 1 - exp(-sampleTime/windowSize)
// 	newRate    = weight*sampleRate + (1-weight)*oldRate
//
// The default values for sampleRate and windowSize (if <= 0) are 100ms and 1s,
// respectively.
func New(sampleRate, windowSize time.Duration) *Monitor {
	if sampleRate = clockRound(sampleRate); sampleRate <= 0 {
		sampleRate = 5 * clockRate
	}
	if windowSize <= 0 {
		windowSize = 1 * time.Second
	}
	now := clock()
	return &Monitor{
		active:  true,
		start:   now,
		rWindow: windowSize.Seconds(),
		sLast:   now,
		sRate:   sampleRate,
		tLast:   now,
	}
}

// Update records the transfer of n bytes and returns n. It should be called
// after each Read/Write operation, even if n is 0.
func (m *Monitor) Update(n int) int {
	m.mu.Lock()
	m.update(n)
	m.mu.Unlock()
	return n
}

// IO is a convenience method intended to wrap io.Reader and io.Writer method
// execution. It calls m.Update(n) and then returns (n, err) unmodified.
func (m *Monitor) IO(n int, err error) (int, error) {
	return m.Update(n), err
}

// Done marks the transfer as finished and prevents any further updates or
// limiting. Instantaneous and current transfer rates drop to 0. Update, IO, and
// Limit methods become NOOPs. It returns the total number of bytes transferred.
func (m *Monitor) Done() int64 {
	m.mu.Lock()
	if now := m.update(0); m.sBytes > 0 {
		m.reset(now)
	}
	m.active = false
	m.tLast = 0
	n := m.bytes
	m.mu.Unlock()
	return n
}

// timeRemLimit is the maximum Status.TimeRem value.
const timeRemLimit = 999*time.Hour + 59*time.Minute + 59*time.Second

// Status represents the current Monitor status. All transfer rates are in bytes
// per second rounded to the nearest byte.
type Status struct {
	Active   bool          // Flag indicating an active transfer
	Start    time.Time     // Transfer start time
	Duration time.Duration // Time period covered by the statistics
	Idle     time.Duration // Time since the last transfer of at least 1 byte
	Bytes    int64         // Total number of bytes transferred
	Samples  int64         // Total number of samples taken
	InstRate int64         // Instantaneous transfer rate
	CurRate  int64         // Current transfer rate (EMA of InstRate)
	AvgRate  int64         // Average transfer rate (Bytes / Duration)
	PeakRate int64         // Maximum instantaneous transfer rate
	BytesRem int64         // Number of bytes remaining in the transfer
	TimeRem  time.Duration // Estimated time to completion
	Progress Percent       // Overall transfer progress
}

// Status returns current transfer status information. The returned value
// becomes static after a call to Done.
func (m *Monitor) Status() Status {
	m.mu.Lock()
	now := m.update(0)
	s := Status{
		Active:   m.active,
		Start:    clockToTime(m.start),
		Duration: m.sLast - m.start,
		Idle:     now - m.tLast,
		Bytes:    m.bytes,
		Samples:  m.samples,
		PeakRate: round(m.rPeak),
		BytesRem: m.tBytes - m.bytes,
		Progress: percentOf(float64(m.bytes), float64(m.tBytes)),
	}
	if s.BytesRem < 0 {
		s.BytesRem = 0
	}
	if s.Duration > 0 {
		rAvg := float64(s.Bytes) / s.Duration.Seconds()
		s.AvgRate = round(rAvg)
		if s.Active {
			s.InstRate = round(m.rSample)
			s.CurRate = round(m.rEMA)
			if s.BytesRem > 0 {
				if tRate := 0.8*m.rEMA + 0.2*rAvg; tRate > 0 {
					ns := float64(s.BytesRem) / tRate * 1e9
					if ns > float64(timeRemLimit) {
						ns = float64(timeRemLimit)
					}
					s.TimeRem = clockRound(time.Duration(ns))
				}
			}
		}
	}
	m.mu.Unlock()
	return s
}

// Limit restricts the instantaneous (per-sample) data flow to rate bytes per
// second. It returns the maximum number of bytes (0 <= n <= want) that may be
// transferred immediately without exceeding the limit. If block == true, the
// call blocks until n > 0. want is returned unmodified if want < 1, rate < 1,
// or the transfer is inactive (after a call to Done).
//
// At least one byte is always allowed to be transferred in any given sampling
// period. Thus, if the sampling rate is 100ms, the lowest achievable flow rate
// is 10 bytes per second.
//
// For usage examples, see the implementation of Reader and Writer in io.go.
func (m *Monitor) Limit(want int, rate int64, block bool) (n int) {
	if want < 1 || rate < 1 {
		return want
	}
	m.mu.Lock()

	// Determine the maximum number of bytes that can be sent in one sample
	limit := round(float64(rate) * m.sRate.Seconds())
	if limit <= 0 {
		limit = 1
	}

	// If block == true, wait until m.sBytes < limit
	if now := m.update(0); block {
		for m.sBytes >= limit && m.active {
			now = m.waitNextSample(now)
		}
	}

	// Make limit <= want (unlimited if the transfer is no longer active)
	if limit -= m.sBytes; limit > int64(want) || !m.active {
		limit = int64(want)
	}
	m.mu.Unlock()

	if limit < 0 {
		limit = 0
	}
	return int(limit)
}

// SetTransferSize specifies the total size of the data transfer, which allows
// the Monitor to calculate the overall progress and time to completion.
func (m *Monitor) SetTransferSize(bytes int64) {
	if bytes < 0 {
		bytes = 0
	}
	m.mu.Lock()
	m.tBytes = bytes
	m.mu.Unlock()
}

// update accumulates the transferred byte count for the current sample until
// clock() - m.sLast >= m.sRate. The monitor status is updated once the current
// sample is done.
func (m *Monitor) update(n int) (now time.Duration) {
	if !m.active {
		return
	}
	if now = clock(); n > 0 {
		m.tLast = now
	}
	m.sBytes += int64(n)
	if sTime := now - m.sLast; sTime >= m.sRate {
		t := sTime.Seconds()
		if m.rSample = float64(m.sBytes) / t; m.rSample > m.rPeak {
			m.rPeak = m.rSample
		}

		// Exponential moving average using a method similar to *nix load
		// average calculation. Longer sampling periods carry greater weight.
		if m.samples > 0 {
			w := math.Exp(-t / m.rWindow)
			m.rEMA = m.rSample + w*(m.rEMA-m.rSample)
		} else {
			m.rEMA = m.rSample
		}
		m.reset(now)
	}
	return
}

// reset clears the current sample state in preparation for the next sample.
func (m *Monitor) reset(sampleTime time.Duration) {
	m.bytes += m.sBytes
	m.samples++
	m.sBytes = 0
	m.sLast = sampleTime
}

// waitNextSample sleeps for the remainder of the current sample. The lock is
// released and reacquired during the actual sleep period, so it's possible for
// the transfer to be inactive when this method returns.
func (m *Monitor) waitNextSample(now time.Duration) time.Duration {
	const minWait = 5 * time.Millisecond
	current := m.sLast

	// sleep until the last sample time changes (ideally, just one iteration)
	for m.sLast == current && m.active {
		d := current + m.sRate - now
		m.mu.Unlock()
		if d < minWait {
			d = minWait
		}
		time.Sleep(d)
		m.mu.Lock()
		now = m.update(0)
	}
	return now
}
