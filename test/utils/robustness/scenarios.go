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
	"net/http"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SignalTriggerComplete is raised by the suite once the scenario's trigger
// action has run. Custom scenarios can gate faults on it via TriggerAfterSignal
// so a fault only fires once reconciliation is actually underway.
const SignalTriggerComplete = "trigger.action.completed"

// ChaosScenario is one cell of the chaos matrix: a named set of faults injected
// while the controller reconciles, plus optional extra verification.
//
// A scenario is self-contained: Faults builds every rule the scenario needs,
// consulting the profile so rules are only produced for sites the controller
// actually exercises. Rules that must demonstrably fire set ExpectTriggered and
// are verified by the suite automatically; Verify is only for assertions beyond
// that (and may be nil).
type ChaosScenario struct {
	Name string

	// Faults returns the fault rules to inject for this scenario. It may return
	// nil when the profile exposes no applicable injection site, in which case
	// the scenario degenerates to a baseline run. The registry is provided for
	// signal-gated conditions (see TriggerAfterSignal).
	Faults func(profile ControllerProfile, registry *FaultRegistry) []FaultRule

	// Verify, if non-nil, runs after convergence for assertions beyond the
	// automatic ExpectTriggered / AssertAllFaultsMatched checks.
	Verify func(profile ControllerProfile, fixture *RobustnessTestFixture)
}

// StandardChaosMatrix returns the default controller-agnostic scenario set. The
// returned slice is fresh on each call, so callers may append custom scenarios
// without affecting other suites (see RobustnessTestSuite.AddScenario).
func StandardChaosMatrix() []ChaosScenario {
	return []ChaosScenario{
		{
			// No faults: verifies the invariants hold under a clean run before
			// trusting them under chaos.
			Name: "BaselineNoFaults",
		},
		{
			Name: "WriteConflicts",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				return rootWriteConflicts(p, TriggerRange(1, 2))
			},
		},
		{
			Name: "CacheSyncLag",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				if !p.HasChildCache() {
					return nil
				}
				return []FaultRule{{
					Name:            "ChildCacheSyncLag",
					Match:           p.ChildCacheMatch(),
					Condition:       TriggerOnOccurrence(1), // lag only the first lookup
					Action:          StaleRead{},
					ExpectTriggered: true,
				}}
			},
		},
		{
			Name: "FlakyAPIServer",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				// Target the child-create path: the root object is typically
				// created by the test via the un-wrapped admin client, so POSTs
				// through the wrapped client are the controller's child creations.
				if !p.CreatesChildren() {
					return nil
				}
				return []FaultRule{{
					Name:      "FlakyWrites",
					Match:     p.ChildCreateMatch(),
					Condition: TriggerProbability(0.30),
					Action:    NewHTTPStatusError(http.StatusInternalServerError, metav1.StatusReasonInternalError, "Internal Server Error: storage write failed"),
					// Not ExpectTriggered: with few creates the 30% coin may
					// legitimately never land. The site itself is still covered
					// by the unmatched-fault guard.
				}}
			},
		},
		{
			Name: "CombinedChaos",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				rules := rootWriteConflicts(p, TriggerOnOccurrence(1))
				if p.HasChildCache() {
					rules = append(rules, FaultRule{
						Name:            "ChildCacheLagCombined",
						Match:           p.ChildCacheMatch(),
						Condition:       TriggerOnOccurrence(1),
						Action:          StaleRead{},
						ExpectTriggered: true,
					})
				}
				return rules
			},
		},
		{
			Name: "ExpectationsTimeout",
			Faults: func(p ControllerProfile, reg *FaultRegistry) []FaultRule {
				var rules []FaultRule
				if p.UsesExpectations {
					// Shift the expectations clock, but only after the trigger
					// action completes, so creation-time expectations are set
					// against real time and then appear expired.
					rules = append(rules, FaultRule{
						Name:            "ShiftExpectationsClock",
						Match:           ClockMatch{Clock: ExpectationsClockName},
						Condition:       TriggerAfterSignal(reg, SignalTriggerComplete),
						Action:          ShiftTime{Offset: 6 * time.Minute},
						ExpectTriggered: true,
					})
				}
				switch {
				case p.HasChildCache() && p.CreatesChildren():
					// Simulate the child informer's watch dying immediately after
					// the controller's create: every cache read (including the
					// last-synced resource version) goes stale for a window
					// starting at the first child POST. Pre-create staleness is
					// harmless — this post-create interleaving is the one that
					// produces duplicate children when the controller acts on the
					// stale view.
					rules = append(rules,
						FaultRule{
							Name:            "ChildCreateObserved",
							Match:           p.ChildCreateMatch(),
							Condition:       TriggerAlways(),
							Action:          nil, // sensor only: opens the window below
							ExpectTriggered: true,
						},
						FaultRule{
							Name:            "StaleCacheWatchDeath",
							Match:           p.ChildCacheMatch(),
							Condition:       TriggerWindowAfterRuleHit(reg, "ChildCreateObserved", time.Second),
							Action:          StaleRead{},
							ExpectTriggered: true,
						})
				case p.HasChildCache():
					// No child creates to key off; fall back to lagging the
					// earliest lookups.
					rules = append(rules, FaultRule{
						Name:            "StaleCacheWatchDeath",
						Match:           p.ChildCacheMatch(),
						Condition:       TriggerRange(1, 3),
						Action:          StaleRead{},
						ExpectTriggered: true,
					})
				}
				return rules
			},
		},
	}
}

// rootWriteConflicts builds 409-conflict rules for the controller's writes to
// the root object and/or its /status subresource, per the profile's write
// declarations. Each produced rule sets ExpectTriggered: the profile promised
// the write happens, so the conflict must demonstrably land.
func rootWriteConflicts(p ControllerProfile, cond FaultCondition) []FaultRule {
	var rules []FaultRule
	if p.WritesRootStatus {
		rules = append(rules, FaultRule{
			Name:            "RootStatusConflict",
			Match:           p.RootStatusWriteMatch(),
			Condition:       cond,
			Action:          NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "Conflict: object status was modified"),
			ExpectTriggered: true,
		})
	}
	if p.WritesRoot {
		rules = append(rules, FaultRule{
			Name:            "RootConflict",
			Match:           p.RootWriteMatch(),
			Condition:       cond,
			Action:          NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "Conflict: object was modified"),
			ExpectTriggered: true,
		})
	}
	return rules
}
