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

// TriggerAfterSignal triggers a fault only once the named signal has been raised at
// least once (see FaultRegistry.Signal). Used to gate faults on a phase of the test,
// e.g. "only after the trigger action has completed".
func TriggerAfterSignal(registry *FaultRegistry, signalName string) FaultCondition {
	return func(matchCount int) bool {
		return registry.SignalCount(signalName) > 0
	}
}

// TriggerWindowAfterRuleHit triggers a fault during a fixed time window that
// opens when the named rule first fires. Pair it with a sensor rule (a rule
// with a nil Action and TriggerAlways) to sequence one fault relative to
// another injection site, e.g. "the child cache goes stale for one second
// starting at the controller's first child create" — the interleaving behind
// classic stale-read bugs, which occurrence-based conditions cannot express
// because they count from the start of the test.
func TriggerWindowAfterRuleHit(registry *FaultRegistry, ruleName string, window time.Duration) FaultCondition {
	return func(matchCount int) bool {
		start := registry.GetFirstHitTime(ruleName)
		if start.IsZero() {
			return false
		}
		return time.Since(start) < window
	}
}

// faultDomain identifies which injection site a fault targets. The domain is
// carried by the FaultMatch (no string parsing), which makes registration-time
// validation and dispatch unambiguous.
type faultDomain int

const (
	domainUnknown faultDomain = iota
	domainTransport
	domainCache
	domainClock
	domainQueue
)

// --- Structured matching. A FaultMatch selects the injection sites a rule applies
// to, by typed fields rather than a flattened string key. Empty string fields are
// wildcards ("match any"), except ClientMatch.Subresource which is matched exactly
// ("" means the main resource, not "any subresource"). ---

// FaultMatch is the structured selector for a fault rule. Each concrete match type
// belongs to exactly one injection domain.
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
	Cache string // cache name, e.g. "pod-cache"
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

// --- Typed verdicts: what an injection site should do when a fault fires. ---
//
// Each injection domain has its own verdict type rather than overloading a shared
// error return. This keeps every site self-contained and makes a mis-targeted
// fault (e.g. a clock fault attached to a transport match) a registration error
// instead of a silent, nonsensical side effect (see validateFaultDomain).

// TransportVerdict tells FaultInjectingTransport how to handle a request.
// The zero value means "pass through to the real transport".
type TransportVerdict struct {
	// Respond, if non-nil, is returned as a synthetic API response instead of
	// forwarding the request to the real transport.
	Respond *HTTPStatusError
	// ConnErr, if non-nil, is returned as a transport-level error (nil response),
	// simulating a refused or dropped connection.
	ConnErr error
}

func (v TransportVerdict) isPass() bool { return v.Respond == nil && v.ConnErr == nil }

// CacheVerdictKind enumerates how a cache lookup should be perturbed.
type CacheVerdictKind int

const (
	CachePass        CacheVerdictKind = iota // no fault; use the real result
	CacheStaleRead                           // behave as if the cache is empty / lagging
	CacheStaleObject                         // return a specific (stale) object
	CacheReturnError                         // return an error
	CacheStaleRV                             // return a specific last-synced resource version
)

// CacheVerdict tells FaultInjectingIndexer how to perturb a lookup. Each indexer
// operation interprets Kind as appropriate for its own return signature.
type CacheVerdict struct {
	Kind   CacheVerdictKind
	Object interface{} // for CacheStaleObject
	Err    error       // for CacheReturnError
	RV     string      // for CacheStaleRV
}

// ClockVerdict tells FaultInjectingClock how to skew time. The zero value is no skew.
type ClockVerdict struct {
	Shift time.Duration // added to the real Now()
}

// --- Domain fault interfaces. A concrete action implements the interface(s) for
// the site(s) it targets; validateFaultDomain rejects an action attached to a
// match whose site it cannot serve. ---

// TransportFault produces a verdict for the REST client transport.
type TransportFault interface{ ApplyTransport() TransportVerdict }

// CacheFault produces a verdict for an informer indexer lookup.
type CacheFault interface{ ApplyCache() CacheVerdict }

// ClockFault produces a verdict for a wrapped clock.
type ClockFault interface{ ApplyClock() ClockVerdict }

// BlockingFault blocks the calling goroutine (a delay or until a signal). It is
// valid at ANY site and does not change the site's return value.
type BlockingFault interface{ Block(ctx context.Context) }

// FaultAction is a marker for fault behaviors. A concrete action implements one or
// more of the domain interfaces above and/or BlockingFault, and is validated
// against its match's domain at registration (see validateFaultDomain). A nil
// Action is permitted (e.g. a rule used only to count matches).
type FaultAction interface{}

// InjectDelay blocks the calling goroutine for a fixed duration (or until ctx is
// cancelled). Valid at any injection site.
type InjectDelay struct {
	Duration time.Duration
}

func (a InjectDelay) Block(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(a.Duration):
	}
}

// BlockUntil blocks the calling goroutine until the channel is closed (or ctx is
// cancelled). Valid at any injection site.
type BlockUntil struct {
	Until <-chan struct{}
}

func (a BlockUntil) Block(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-a.Until:
	}
}

// FaultRule declares a fault condition and side effect bound to a set of injection
// sites selected by Match.
type FaultRule struct {
	Name      string
	Match     FaultMatch
	Condition FaultCondition
	Action    FaultAction

	// Optional, when true, exempts this rule from the "every fault must match an
	// injection site" check (see FaultRegistry.UnmatchedRules). Use it for faults
	// that are only conditionally reachable (e.g. gated behind a feature flag).
	// Leave it false for real faults so a mis-targeted Match surfaces as a test
	// failure instead of a silent no-op.
	Optional bool

	// ExpectTriggered, when true, declares that this fault must actually fire
	// (condition satisfied, action applied) at least once during the scenario,
	// not merely match a site. The suite verifies this after convergence (see
	// FaultRegistry.ExpectedRulesNotTriggered), replacing hand-written per-rule
	// hit-count assertions. Leave it false for probabilistic faults that may
	// legitimately never fire.
	ExpectTriggered bool

	// Track matching metrics atomically
	matchCount int32
	hitCount   int32
	firstHit   int64 // Unix nanoseconds, 0 means not hit
}

// FaultRegistry manages active fault rules and orchestrates trigger evaluations.
type FaultRegistry struct {
	mu    sync.RWMutex
	rules map[faultDomain][]*FaultRule

	signalsMu sync.Mutex
	signals   map[string]int
}

// NewFaultRegistry creates a thread-safe FaultRegistry.
func NewFaultRegistry() *FaultRegistry {
	return &FaultRegistry{
		rules:   make(map[faultDomain][]*FaultRule),
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
	r.rules[d] = append(r.rules[d], &rule)
}

// Signal raises a named signal, incrementing its count. Used together with
// TriggerAfterSignal to gate faults on a test phase.
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

// fire evaluates every rule in the given domain whose match accepts the facts
// (via pred): it increments the rule's match counter, checks its condition, and
// for a triggered rule runs any blocking effect and then invokes apply with the
// action so the caller can extract a domain verdict. apply may be nil for
// side-effect-only sites (the work queue).
func (r *FaultRegistry) fire(ctx context.Context, domain faultDomain, pred func(FaultMatch) bool, apply func(action FaultAction)) {
	r.mu.RLock()
	rules := r.rules[domain]
	r.mu.RUnlock()

	for _, rule := range rules {
		if !pred(rule.Match) {
			continue
		}
		match := atomic.AddInt32(&rule.matchCount, 1)
		if rule.Condition == nil || !rule.Condition(int(match)) {
			continue
		}
		atomic.CompareAndSwapInt64(&rule.firstHit, 0, time.Now().UnixNano())
		atomic.AddInt32(&rule.hitCount, 1)
		if rule.Action == nil {
			continue
		}
		if b, ok := rule.Action.(BlockingFault); ok {
			b.Block(ctx)
		}
		if apply != nil {
			apply(rule.Action)
		}
	}
}

// ResolveTransport returns the verdict of the first triggered TransportFault that
// matches facts. The zero value means "pass through to the real transport".
func (r *FaultRegistry) ResolveTransport(ctx context.Context, facts ClientFacts) TransportVerdict {
	var v TransportVerdict
	r.fire(ctx, domainTransport, func(m FaultMatch) bool {
		cm, ok := m.(ClientMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if !v.isPass() {
			return
		}
		if tf, ok := action.(TransportFault); ok {
			v = tf.ApplyTransport()
		}
	})
	return v
}

// ResolveCache returns the verdict of the first triggered CacheFault that matches
// facts. CachePass means "use the real result".
func (r *FaultRegistry) ResolveCache(ctx context.Context, facts CacheFacts) CacheVerdict {
	v := CacheVerdict{Kind: CachePass}
	r.fire(ctx, domainCache, func(m FaultMatch) bool {
		cm, ok := m.(CacheMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if v.Kind != CachePass {
			return
		}
		if cf, ok := action.(CacheFault); ok {
			v = cf.ApplyCache()
		}
	})
	return v
}

// ResolveClock returns the total time shift contributed by all triggered
// ClockFaults that match facts.
func (r *FaultRegistry) ResolveClock(ctx context.Context, facts ClockFacts) time.Duration {
	var shift time.Duration
	r.fire(ctx, domainClock, func(m FaultMatch) bool {
		cm, ok := m.(ClockMatch)
		return ok && cm.matches(facts)
	}, func(action FaultAction) {
		if cf, ok := action.(ClockFault); ok {
			shift += cf.ApplyClock().Shift
		}
	})
	return shift
}

// ResolveQueue fires any blocking faults registered on work queue operations that
// match facts. Work queue faults have no verdict; they only delay or block.
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
		for _, rule := range ruleList {
			if rule.Name == ruleName {
				total += int(atomic.LoadInt32(&rule.hitCount))
			}
		}
	}
	return total
}

// GetFirstHitTime returns the timestamp of the first time a rule with the given name triggered.
// Returns the zero time if the rule has not triggered.
func (r *FaultRegistry) GetFirstHitTime(ruleName string) time.Time {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var earliest int64
	for _, ruleList := range r.rules {
		for _, rule := range ruleList {
			if rule.Name == ruleName {
				hit := atomic.LoadInt64(&rule.firstHit)
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

// UnmatchedRules returns the names of all registered non-optional rules whose
// match never accepted a real injection site (matchCount == 0).
//
// A rule that never matches injects nothing: the test goes green while asserting
// nothing, which is the most dangerous failure mode for a fault-injection
// harness. The usual cause is a Match that doesn't line up with a real injection
// site (a mis-pluralized resource, a subresource that is never written, or
// traffic that flows through an un-wrapped client). Callers should treat a
// non-empty result as a test failure. Rules marked Optional are skipped.
//
// Note this checks matchCount, not hitCount: a probabilistic fault whose site was
// exercised but whose Condition happened not to fire is still considered matched.
func (r *FaultRegistry) UnmatchedRules() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var unmatched []string
	for _, ruleList := range r.rules {
		for _, rule := range ruleList {
			if rule.Optional {
				continue
			}
			if atomic.LoadInt32(&rule.matchCount) == 0 {
				unmatched = append(unmatched, rule.Name)
			}
		}
	}
	return unmatched
}

// ExpectedRulesNotTriggered returns the names of all rules registered with
// ExpectTriggered whose action never actually fired (hitCount == 0).
//
// This complements UnmatchedRules: matching proves the rule lined up with a real
// injection site, while this proves the fault was actually delivered. A scenario
// that declares ExpectTriggered on a rule gets its hit-count assertion for free.
func (r *FaultRegistry) ExpectedRulesNotTriggered() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var silent []string
	for _, ruleList := range r.rules {
		for _, rule := range ruleList {
			if !rule.ExpectTriggered {
				continue
			}
			if atomic.LoadInt32(&rule.hitCount) == 0 {
				silent = append(silent, rule.Name)
			}
		}
	}
	return silent
}

// validateFaultDomain checks that a rule's action can actually be served at the
// injection site its match selects, so a mis-targeted fault (e.g. a clock fault on
// a transport match) fails loudly at registration instead of silently doing
// nothing useful. A nil action and BlockingFault actions are valid for any match.
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
