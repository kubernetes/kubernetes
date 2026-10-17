/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package robustness

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// FaultCondition evaluates whether a matched hook point should execute the fault action.
// matchCount represents how many times this specific rule's matcher has fired.
type FaultCondition func(matchCount int) bool

// Standard Conditions

// TriggerOnOccurrence triggers the fault only on the N-th match (1-indexed).
func TriggerOnOccurrence(n int) FaultCondition {
	return func(matchCount int) bool {
		return matchCount == n
	}
}

// TriggerRange triggers the fault between the start and end match counts (inclusive).
func TriggerRange(start, end int) FaultCondition {
	return func(matchCount int) bool {
		return matchCount >= start && matchCount <= end
	}
}

// TriggerProbability triggers the fault randomly with a probability between 0.0 and 1.0.
func TriggerProbability(prob float64) FaultCondition {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	var mu sync.Mutex
	return func(matchCount int) bool {
		mu.Lock()
		defer mu.Unlock()
		return r.Float64() < prob
	}
}

// TriggerAlways triggers the fault on every match.
func TriggerAlways() FaultCondition {
	return func(matchCount int) bool {
		return true
	}
}

// TriggerAfterSignal triggers once the named signal has been raised (see FaultRegistry.Signal).
func TriggerAfterSignal(registry *FaultRegistry, signalName string) FaultCondition {
	return func(matchCount int) bool {
		return registry.SignalCount(signalName) > 0
	}
}

// TriggerWindowAfterRuleHit triggers during a fixed duration starting when ruleName
// first fires. Pair with a sensor rule (nil Action, TriggerAlways) to sequence one
// fault relative to another site (e.g. stale cache reads starting at the first POST).
func TriggerWindowAfterRuleHit(registry *FaultRegistry, ruleName string, window time.Duration) FaultCondition {
	return func(matchCount int) bool {
		start := registry.GetFirstHitTime(ruleName)
		if start.IsZero() {
			return false
		}
		return time.Since(start) < window
	}
}

type faultDomain int

const (
	domainUnknown faultDomain = iota
	domainTransport
	domainCache
	domainClock
	domainQueue
)

// FaultMatch selects injection sites for a fault rule. Empty string fields are
// wildcards, except ClientMatch.Subresource which matches exact ("" = main resource).
type FaultMatch interface {
	domain() faultDomain
}

func wildcard(matcher, fact string) bool { return matcher == "" || matcher == fact }

// ClientFacts describes a REST request seen by the transport.
type ClientFacts struct {
	Verb        string // HTTP method, e.g. "PUT", "POST"
	Group       string // API group, e.g. "apps" ("" for the core group)
	Resource    string // plural resource, e.g. "daemonsets"
	Subresource string // e.g. "status" ("" for the main resource)
	Namespace   string
	Name        string
}

// ClientMatch matches REST requests. Empty fields are wildcards, except
// Subresource which is matched exactly ("" = main resource only).
type ClientMatch struct {
	Verb        string
	Group       string
	Resource    string
	Subresource string
	Namespace   string
	Name        string
}

func (ClientMatch) domain() faultDomain { return domainTransport }

func (m ClientMatch) matches(f ClientFacts) bool {
	return wildcard(m.Verb, f.Verb) &&
		wildcard(m.Group, f.Group) &&
		wildcard(m.Resource, f.Resource) &&
		m.Subresource == f.Subresource &&
		wildcard(m.Namespace, f.Namespace) &&
		wildcard(m.Name, f.Name)
}

// CacheFacts describes an informer cache lookup.
type CacheFacts struct {
	Cache string // plural resource name, e.g. "pods"
	Op    string // "get", "list", "by-index", "last-sync-rv"
	Key   string // object key for get/by-index
}

// CacheMatch matches informer cache lookups. Empty fields are wildcards.
type CacheMatch struct {
	Cache string
	Op    string
	Key   string
}

func (CacheMatch) domain() faultDomain { return domainCache }

func (m CacheMatch) matches(f CacheFacts) bool {
	return wildcard(m.Cache, f.Cache) && wildcard(m.Op, f.Op) && wildcard(m.Key, f.Key)
}

// ClockFacts describes a clock read.
type ClockFacts struct {
	Clock string // clock name, e.g. "expectations"
}

// ClockMatch matches clock reads. An empty Clock matches any clock.
type ClockMatch struct {
	Clock string
}

func (ClockMatch) domain() faultDomain { return domainClock }

func (m ClockMatch) matches(f ClockFacts) bool { return wildcard(m.Clock, f.Clock) }

// QueueFacts describes a work queue operation.
type QueueFacts struct {
	Queue string // queue name
	Op    string // "get", "add"
}

// QueueMatch matches work queue operations. Empty fields are wildcards.
type QueueMatch struct {
	Queue string
	Op    string
}

func (QueueMatch) domain() faultDomain { return domainQueue }

func (m QueueMatch) matches(f QueueFacts) bool {
	return wildcard(m.Queue, f.Queue) && wildcard(m.Op, f.Op)
}

// TransportFault returns a synthetic API error for the REST transport (nil passes through).
type TransportFault interface{ ApplyTransport() *HTTPStatusError }

// CacheFault reports whether an informer cache lookup should simulate a stale read.
type CacheFault interface{ ApplyCache() (stale bool) }

// ClockFault returns the time offset to add to a wrapped clock's Now().
type ClockFault interface{ ApplyClock() time.Duration }

// BlockingFault blocks the calling goroutine and is valid at any injection site.
type BlockingFault interface{ Block(ctx context.Context) }

// FaultAction is implemented by domain fault actions and BlockingFault. A nil
// Action is permitted for sensor rules that only record match/hit timestamps.
type FaultAction any

// InjectDelay blocks the calling goroutine for Duration (or until ctx is cancelled).
type InjectDelay struct {
	Duration time.Duration
}

func (a InjectDelay) Block(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(a.Duration):
	}
}

// FaultRule binds a Match selector and trigger Condition to a FaultAction.
type FaultRule struct {
	Name      string
	Match     FaultMatch
	Condition FaultCondition
	Action    FaultAction

	// Optional exempts this rule from UnmatchedRules checks (for feature-gated sites).
	Optional bool

	// ExpectTriggered requires this rule's action to fire at least once during the scenario.
	ExpectTriggered bool
}

// activeRule tracks a registered rule and its runtime execution metrics atomically.
type activeRule struct {
	rule FaultRule

	matchCount atomic.Int32
	hitCount   atomic.Int32
	firstHit   atomic.Int64 // Unix nanoseconds, 0 means not hit
}

// FaultRegistry manages active fault rules and orchestrates trigger evaluations.
type FaultRegistry struct {
	mu    sync.RWMutex
	rules map[faultDomain][]*activeRule

	signalsMu sync.Mutex
	signals   map[string]int
}

// NewFaultRegistry creates a thread-safe FaultRegistry.
func NewFaultRegistry() *FaultRegistry {
	return &FaultRegistry{
		rules:   make(map[faultDomain][]*activeRule),
		signals: make(map[string]int),
	}
}

// Register adds a new FaultRule to the registry, bucketed by its match domain.
func (r *FaultRegistry) Register(rule FaultRule) {
	d := domainUnknown
	if rule.Match != nil {
		d = rule.Match.domain()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rules[d] = append(r.rules[d], &activeRule{rule: rule})
}

// Signal raises a named signal, incrementing its count.
func (r *FaultRegistry) Signal(name string) {
	r.signalsMu.Lock()
	defer r.signalsMu.Unlock()
	r.signals[name]++
}

// SignalCount returns how many times the named signal has been raised.
func (r *FaultRegistry) SignalCount(name string) int {
	r.signalsMu.Lock()
	defer r.signalsMu.Unlock()
	return r.signals[name]
}

func (r *FaultRegistry) fire(ctx context.Context, domain faultDomain, pred func(FaultMatch) bool, apply func(action FaultAction)) {
	r.mu.RLock()
	rules := r.rules[domain]
	r.mu.RUnlock()

	for _, ar := range rules {
		if !pred(ar.rule.Match) {
			continue
		}
		match := ar.matchCount.Add(1)
		if ar.rule.Condition == nil || !ar.rule.Condition(int(match)) {
			continue
		}
		ar.firstHit.CompareAndSwap(0, time.Now().UnixNano())
		ar.hitCount.Add(1)
		if ar.rule.Action == nil {
			continue
		}
		if b, ok := ar.rule.Action.(BlockingFault); ok {
			b.Block(ctx)
		}
		if apply != nil {
			apply(ar.rule.Action)
		}
	}
}

// ResolveTransport returns the first triggered HTTPStatusError matching facts, or nil to pass through.
func (r *FaultRegistry) ResolveTransport(ctx context.Context, facts ClientFacts) *HTTPStatusError {
	var resp *HTTPStatusError
	r.fire(ctx, domainTransport, func(m FaultMatch) bool {
		cm, ok := m.(ClientMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if resp != nil {
			return
		}
		if tf, ok := action.(TransportFault); ok {
			resp = tf.ApplyTransport()
		}
	})
	return resp
}

// ResolveCache reports whether any triggered CacheFault matching facts requests a stale read.
func (r *FaultRegistry) ResolveCache(ctx context.Context, facts CacheFacts) bool {
	var stale bool
	r.fire(ctx, domainCache, func(m FaultMatch) bool {
		cm, ok := m.(CacheMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if stale {
			return
		}
		if cf, ok := action.(CacheFault); ok {
			stale = cf.ApplyCache()
		}
	})
	return stale
}

// ResolveClock returns the total time shift from all triggered ClockFaults matching facts.
func (r *FaultRegistry) ResolveClock(ctx context.Context, facts ClockFacts) time.Duration {
	var shift time.Duration
	r.fire(ctx, domainClock, func(m FaultMatch) bool {
		cm, ok := m.(ClockMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if cf, ok := action.(ClockFault); ok {
			shift += cf.ApplyClock()
		}
	})
	return shift
}

// ResolveQueue runs any blocking faults registered on matching work queue operations.
func (r *FaultRegistry) ResolveQueue(ctx context.Context, facts QueueFacts) {
	r.fire(ctx, domainQueue, func(m FaultMatch) bool {
		qm, ok := m.(QueueMatch)
		return ok && qm.matches(facts)
	}, nil)
}

// GetHitCount returns the total number of times faults with the specified ruleName were executed.
func (r *FaultRegistry) GetHitCount(ruleName string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var total int
	for _, ruleList := range r.rules {
		for _, ar := range ruleList {
			if ar.rule.Name == ruleName {
				total += int(ar.hitCount.Load())
			}
		}
	}
	return total
}

// GetFirstHitTime returns the timestamp of the first time ruleName triggered, or the zero time.
func (r *FaultRegistry) GetFirstHitTime(ruleName string) time.Time {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var earliest int64
	for _, ruleList := range r.rules {
		for _, ar := range ruleList {
			if ar.rule.Name == ruleName {
				hit := ar.firstHit.Load()
				if hit > 0 && (earliest == 0 || hit < earliest) {
					earliest = hit
				}
			}
		}
	}
	if earliest == 0 {
		return time.Time{}
	}
	return time.Unix(0, earliest)
}

// UnmatchedRules returns the names of non-optional rules whose Match never matched an injection site.
func (r *FaultRegistry) UnmatchedRules() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var unmatched []string
	for _, ruleList := range r.rules {
		for _, ar := range ruleList {
			if ar.rule.Optional {
				continue
			}
			if ar.matchCount.Load() == 0 {
				unmatched = append(unmatched, ar.rule.Name)
			}
		}
	}
	return unmatched
}

// ExpectedRulesNotTriggered returns the names of ExpectTriggered rules that never fired.
func (r *FaultRegistry) ExpectedRulesNotTriggered() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var silent []string
	for _, ruleList := range r.rules {
		for _, ar := range ruleList {
			if !ar.rule.ExpectTriggered {
				continue
			}
			if ar.hitCount.Load() == 0 {
				silent = append(silent, ar.rule.Name)
			}
		}
	}
	return silent
}

// validateFaultDomain ensures a rule's action is compatible with its match domain.
func validateFaultDomain(rule FaultRule) error {
	if rule.Match == nil {
		return fmt.Errorf("fault rule %q has no Match", rule.Name)
	}
	if rule.Action == nil {
		return nil
	}
	if _, ok := rule.Action.(BlockingFault); ok {
		return nil // blocking faults are valid at any site
	}
	switch rule.Match.domain() {
	case domainTransport:
		if _, ok := rule.Action.(TransportFault); !ok {
			return fmt.Errorf("ClientMatch targets the REST transport but action %T is not a TransportFault", rule.Action)
		}
	case domainCache:
		if _, ok := rule.Action.(CacheFault); !ok {
			return fmt.Errorf("CacheMatch targets an informer cache but action %T is not a CacheFault", rule.Action)
		}
	case domainClock:
		if _, ok := rule.Action.(ClockFault); !ok {
			return fmt.Errorf("ClockMatch targets a clock but action %T is not a ClockFault", rule.Action)
		}
	case domainQueue:
		return fmt.Errorf("QueueMatch only supports BlockingFault actions, but got %T", rule.Action)
	}
	return nil
}
