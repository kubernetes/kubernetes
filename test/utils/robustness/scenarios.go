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
	"k8s.io/kubernetes/pkg/controller"
)

// SignalTriggerComplete is raised once the scenario's trigger action completes.
const SignalTriggerComplete = "trigger.action.completed"

// ChaosScenario is one cell of the chaos matrix: a named set of faults derived
// from the ControllerProfile, plus optional post-convergence verification.
type ChaosScenario struct {
	Name string

	// Faults returns the fault rules to inject for this scenario, or nil when
	// the profile exposes no applicable injection site.
	Faults func(profile ControllerProfile, registry *FaultRegistry) []FaultRule

	// Verify, if non-nil, runs after convergence for extra assertions.
	Verify func(profile ControllerProfile, fixture *RobustnessTestFixture)
}

// StandardChaosMatrix returns the default controller-agnostic scenario set.
func StandardChaosMatrix() []ChaosScenario {
	return []ChaosScenario{
		{
			Name: "BaselineNoFaults",
		},
		{
			Name: "WriteConflicts",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				return rootWriteConflicts(p, TriggerRange(1, p.writeFaultBurst()))
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
					Condition:       TriggerOnOccurrence(1),
					Action:          StaleRead{},
					ExpectTriggered: true,
				}}
			},
		},
		{
			Name: "FlakyAPIServer",
			Faults: func(p ControllerProfile, reg *FaultRegistry) []FaultRule {
				// First write always fails (so ExpectTriggered holds), then 30%.
				flaky := func() FaultCondition {
					return TriggerAnyOf(TriggerOnOccurrence(1), TriggerProbability(reg, 0.30))
				}
				switch {
				case p.CreatesChildren():
					return []FaultRule{{
						Name:            "FlakyChildWrites",
						Match:           p.ChildCreateMatch(),
						Condition:       flaky(),
						Action:          NewHTTPStatusError(http.StatusInternalServerError, metav1.StatusReasonInternalError, "Internal Server Error: storage write failed"),
						ExpectTriggered: true,
					}}
				case p.WritesRootStatus:
					return []FaultRule{{
						Name:            "FlakyRootStatusWrites",
						Match:           p.RootStatusWriteMatch(),
						Condition:       flaky(),
						Action:          NewHTTPStatusError(http.StatusInternalServerError, metav1.StatusReasonInternalError, "Internal Server Error: status write failed"),
						ExpectTriggered: true,
					}}
				case p.WritesRoot:
					return []FaultRule{{
						Name:            "FlakyRootWrites",
						Match:           p.RootWriteMatch(),
						Condition:       flaky(),
						Action:          NewHTTPStatusError(http.StatusInternalServerError, metav1.StatusReasonInternalError, "Internal Server Error: object write failed"),
						ExpectTriggered: true,
					}}
				default:
					return nil
				}
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
			// Stall the root watch at the first root event so the controller's
			// cache stays stale for WatchStall while the API server moves on.
			Name: "StaleRootWatch",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				if p.WatchStall == 0 {
					return nil
				}
				return []FaultRule{{
					Name:            "StallRootWatch",
					Match:           p.RootWatchMatch(),
					Condition:       TriggerOnOccurrence(1),
					Action:          InjectDelay{Duration: p.WatchStall},
					ExpectTriggered: true,
				}}
			},
		},
		{
			// Same for the child watch: the controller's own writes become
			// invisible to it for WatchStall.
			Name: "StaleChildWatch",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				if p.WatchStall == 0 || !p.HasChildCache() {
					return nil
				}
				return []FaultRule{{
					Name:            "StallChildWatch",
					Match:           p.ChildWatchMatch(),
					Condition:       TriggerOnOccurrence(1),
					Action:          InjectDelay{Duration: p.WatchStall},
					ExpectTriggered: true,
				}}
			},
		},
		{
			Name: "ExpectationsTimeout",
			Faults: func(p ControllerProfile, reg *FaultRegistry) []FaultRule {
				var rules []FaultRule
				createsChildren := p.HasChildCache() && p.CreatesChildren()
				switch {
				case createsChildren && p.WatchStall > 0:
					// The child watch dies at the first child event, i.e. the
					// controller's own create. Its expectation is never fulfilled
					// and its cache stays stale for WatchStall; with the clock
					// shifted below, a controller that trusts expired expectations
					// re-syncs against that stale cache (and, e.g., double-creates).
					rules = append(rules,
						FaultRule{
							Name:            "ChildCreateObserved",
							Match:           p.ChildCreateMatch(),
							Condition:       TriggerAlways(),
							Action:          nil,
							ExpectTriggered: true,
						},
						FaultRule{
							Name:            "ChildWatchDeath",
							Match:           p.ChildWatchMatch(),
							Condition:       TriggerOnOccurrence(1),
							Action:          InjectDelay{Duration: p.WatchStall},
							ExpectTriggered: true,
						})
				case createsChildren:
					// Without WatchStall, approximate the dead watch with stale
					// reads for a second after the first child POST.
					rules = append(rules,
						FaultRule{
							Name:            "ChildCreateObserved",
							Match:           p.ChildCreateMatch(),
							Condition:       TriggerAlways(),
							Action:          nil,
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
					rules = append(rules, FaultRule{
						Name:            "StaleCacheWatchDeath",
						Match:           p.ChildCacheMatch(),
						Condition:       TriggerRange(1, 3),
						Action:          StaleRead{},
						ExpectTriggered: true,
					})
				}
				if p.UsesExpectations {
					// The shift must start only after the controller has recorded
					// its expectations; otherwise SetExpectations stamps the shifted
					// time and isExpired (Since on the same clock) sees ~0 elapsed.
					gate := TriggerAfterSignal(reg, SignalTriggerComplete)
					if createsChildren {
						gate = TriggerAfterRuleHit(reg, "ChildCreateObserved")
					}
					rules = append(rules, FaultRule{
						Name:            "ShiftExpectationsClock",
						Match:           ClockMatch{Clock: ExpectationsClockName},
						Condition:       gate,
						Action:          ShiftTime{Offset: controller.ExpectationsTimeout + time.Minute},
						ExpectTriggered: true,
					})
				}
				return rules
			},
		},
	}
}

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
