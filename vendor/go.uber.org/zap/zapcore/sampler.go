// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zapcore

import (
	"time"

	"go.uber.org/atomic"
)

const (
	_numLevels        = _maxLevel - _minLevel + 1
	_countersPerLevel = 4096
)

type counter struct {
	resetAt atomic.Int64
	counter atomic.Uint64
}

type counters [_numLevels][_countersPerLevel]counter

func newCounters() *counters {
	return &counters{}
}

func (cs *counters) get(lvl Level, key string) *counter {
	i := lvl - _minLevel
	j := fnv32a(key) % _countersPerLevel
	return &cs[i][j]
}

// fnv32a, adapted from "hash/fnv", but without a []byte(string) alloc
func fnv32a(s string) uint32 {
	const (
		offset32 = 2166136261
		prime32  = 16777619
	)
	hash := uint32(offset32)
	for i := 0; i < len(s); i++ {
		hash ^= uint32(s[i])
		hash *= prime32
	}
	return hash
}

func (c *counter) IncCheckReset(t time.Time, tick time.Duration) uint64 {
	tn := t.UnixNano()
	resetAfter := c.resetAt.Load()
	if resetAfter > tn {
		return c.counter.Inc()
	}

	c.counter.Store(1)

	newResetAfter := tn + tick.Nanoseconds()
	if !c.resetAt.CAS(resetAfter, newResetAfter) {
		// We raced with another goroutine trying to reset, and it also reset
		// the counter to 1, so we need to reincrement the counter.
		return c.counter.Inc()
	}

	return 1
}

// SamplingDecision is a decision represented as a bit field made by sampler.
// More decisions may be added in the future.
type SamplingDecision uint32

const (
	// LogDropped indicates that the Sampler dropped a log entry.
	LogDropped SamplingDecision = 1 << iota
	// LogSampled indicates that the Sampler sampled a log entry.
	LogSampled
)

// optionFunc wraps a func so it satisfies the SamplerOption interface.
type optionFunc func(*sampler)

func (f optionFunc) apply(s *sampler) {
	f(s)
}

// SamplerOption configures a Sampler.
type SamplerOption interface {
	apply(*sampler)
}

// nopSamplingHook is the default hook used by sampler.
func nopSamplingHook(Entry, SamplingDecision) {}

// SamplerHook registers a function  which will be called when Sampler makes a
// decision.
//
// This hook may be used to get visibility into the performance of the sampler.
// For example, use it to track metrics of dropped versus sampled logs.
//
//  var dropped atomic.Int64
//  zapcore.SamplerHook(func(ent zapcore.Entry, dec zapcore.SamplingDecision) {
//    if dec&zapcore.LogDropped > 0 {
//      dropped.Inc()
//    }
//  })
func SamplerHook(hook func(entry Entry, dec SamplingDecision)) SamplerOption {
	return optionFunc(func(s *sampler) {
		s.hook = hook
	})
}

// NewSamplerWithOptions creates a Core that samples incoming entries, which
// caps the CPU and I/O load of logging while attempting to preserve a
// representative subset of your logs.
//
// Zap samples by logging the first N entries with a given level and message
// each tick. If more Entries with the same level and message are seen during
// the same interval, every Mth message is logged and the rest are dropped.
//
// Sampler can be configured to report sampling decisions with the SamplerHook
// option.
//
// Keep in mind that zap's sampling implementation is optimized for speed over
// absolute precision; under load, each tick may be slightly over- or
// under-sampled.
func NewSamplerWithOptions(core Core, tick time.Duration, first, thereafter int, opts ...SamplerOption) Core {
	s := &sampler{
		Core:       core,
		tick:       tick,
		counts:     newCounters(),
		first:      uint64(first),
		thereafter: uint64(thereafter),
		hook:       nopSamplingHook,
	}
	for _, opt := range opts {
		opt.apply(s)
	}

	return s
}

type sampler struct {
	Core

	counts            *counters
	tick              time.Duration
	first, thereafter uint64
	hook              func(Entry, SamplingDecision)
}

// NewSampler creates a Core that samples incoming entries, which
// caps the CPU and I/O load of logging while attempting to preserve a
// representative subset of your logs.
//
// Zap samples by logging the first N entries with a given level and message
// each tick. If more Entries with the same level and message are seen during
// the same interval, every Mth message is logged and the rest are dropped.
//
// Keep in mind that zap's sampling implementation is optimized for speed over
// absolute precision; under load, each tick may be slightly over- or
// under-sampled.
//
// Deprecated: use NewSamplerWithOptions.
func NewSampler(core Core, tick time.Duration, first, thereafter int) Core {
	return NewSamplerWithOptions(core, tick, first, thereafter)
}

func (s *sampler) With(fields []Field) Core {
	return &sampler{
		Core:       s.Core.With(fields),
		tick:       s.tick,
		counts:     s.counts,
		first:      s.first,
		thereafter: s.thereafter,
		hook:       s.hook,
	}
}

func (s *sampler) Check(ent Entry, ce *CheckedEntry) *CheckedEntry {
	if !s.Enabled(ent.Level) {
		return ce
	}

	counter := s.counts.get(ent.Level, ent.Message)
	n := counter.IncCheckReset(ent.Time, s.tick)
	if n > s.first && (n-s.first)%s.thereafter != 0 {
		s.hook(ent, LogDropped)
		return ce
	}
	s.hook(ent, LogSampled)
	return s.Core.Check(ent, ce)
}
