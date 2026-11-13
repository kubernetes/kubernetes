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
			Faults: func(p ControllerProfile, reg *FaultRegistry) []FaultRule {
				if !p.HasChildCache() {
					return nil
				}
				return childCacheLag(p, reg, "ChildCacheSyncLag")
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
			Faults: func(p ControllerProfile, reg *FaultRegistry) []FaultRule {
				rules := rootWriteConflicts(p, TriggerOnOccurrence(1))
				if p.HasChildCache() {
					rules = append(rules, childCacheLag(p, reg, "ChildCacheLagCombined")...)
				}
				return rules
			},
		},
		{
			// Stall the root watch at the first root event so the controller's
			// cache stays stale for StaleCacheWindow while the API server moves on.
			Name: "StaleRootWatch",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				return []FaultRule{{
					Name:            "StallRootWatch",
					Match:           p.RootWatchMatch(),
					Condition:       TriggerOnOccurrence(1),
					Action:          InjectDelay{Duration: p.staleCacheWindow()},
					ExpectTriggered: true,
				}}
			},
		},
		{
			// Same for the child watch: the controller's own writes become
			// invisible to it for StaleCacheWindow.
			Name: "StaleChildWatch",
			Faults: func(p ControllerProfile, _ *FaultRegistry) []FaultRule {
				if !p.HasChildCache() {
					return nil
				}
				return []FaultRule{{
					Name:            "StallChildWatch",
					Match:           p.ChildWatchMatch(),
					Condition:       TriggerOnOccurrence(1),
					Action:          InjectDelay{Duration: p.staleCacheWindow()},
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
				case createsChildren:
					// Kill the child watch at the controller's own create. The
					// expectation stays unfulfilled and the cache stale for
					// StaleCacheWindow, so with the clock shifted below a controller
					// that trusts expired expectations re-syncs against a stale
					// cache and may double-create.
					rules = append(rules,
						childCreateSensor(p),
						FaultRule{
							Name:            "ChildWatchDeath",
							Match:           p.ChildWatchMatch(),
							Condition:       TriggerOnOccurrence(1),
							Action:          InjectDelay{Duration: p.staleCacheWindow()},
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
					// Shift only after expectations are recorded. Otherwise
					// SetExpectations stamps the shifted time and nothing expires.
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

// childCreateSensor records child POSTs without changing them, so other rules
// can key off the first create.
func childCreateSensor(p ControllerProfile) FaultRule {
	return FaultRule{
		Name:            "ChildCreateObserved",
		Match:           p.ChildCreateMatch(),
		Condition:       TriggerAlways(),
		Action:          nil,
		ExpectTriggered: true,
	}
}

// childCacheLag returns stale child reads for one second after the first child
// create. Before that the cache is empty either way. Controllers that do not
// create children get the lag on their first lookup.
func childCacheLag(p ControllerProfile, reg *FaultRegistry, name string) []FaultRule {
	if !p.CreatesChildren() {
		return []FaultRule{{
			Name:            name,
			Match:           p.ChildCacheMatch(),
			Condition:       TriggerOnOccurrence(1),
			Action:          StaleRead{},
			ExpectTriggered: true,
		}}
	}
	return []FaultRule{
		childCreateSensor(p),
		{
			Name:            name,
			Match:           p.ChildCacheMatch(),
			Condition:       TriggerWindowAfterRuleHit(reg, "ChildCreateObserved", time.Second),
			Action:          StaleRead{},
			ExpectTriggered: true,
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
