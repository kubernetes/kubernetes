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
	"strings"
	"time"

	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
)

// ControllerSetupFn initializes and runs the controller under test using the
// fixture's wrapped clients.
type ControllerSetupFn func(tCtx ktesting.TContext, fixture *RobustnessTestFixture)

// ScenarioActionFn executes the action that triggers reconciliation.
type ScenarioActionFn func(ctx context.Context, fixture *RobustnessTestFixture) error

// RobustnessTestSuite runs a controller's trigger action and invariants across
// every scenario of a chaos matrix.
type RobustnessTestSuite struct {
	tCtx    ktesting.TContext
	profile ControllerProfile
	matrix  []ChaosScenario

	controllerSetup      ControllerSetupFn
	scenarioAction       ScenarioActionFn
	continuousInvariants []NamedInvariant
	livenessInvariants   []NamedInvariant
	livenessTimeout      time.Duration

	checkWhenSettled bool
	quietWindow      time.Duration

	scenarioFaults map[string][]func(ControllerProfile, *FaultRegistry) []FaultRule
	scenarioVerify map[string][]func(ControllerProfile, *RobustnessTestFixture)
}

// NewTestSuite creates a chaos-matrix runner for the controller described by
// profile, preloaded with the standard scenario matrix.
func NewTestSuite(tCtx ktesting.TContext, profile ControllerProfile) *RobustnessTestSuite {
	if err := profile.validate(); err != nil {
		tCtx.Fatalf("invalid ControllerProfile: %v", err)
	}
	return &RobustnessTestSuite{
		tCtx:            tCtx,
		profile:         profile,
		matrix:          StandardChaosMatrix(),
		livenessTimeout: 15 * time.Second,
	}
}

// Profile returns the declared controller profile.
func (s *RobustnessTestSuite) Profile() ControllerProfile {
	return s.profile
}

// AddScenario appends a custom scenario to the matrix.
func (s *RobustnessTestSuite) AddScenario(scenario ChaosScenario) {
	s.matrix = append(s.matrix, scenario)
}

// AddScenarioFaults registers extra fault rules (typically sensors) for the
// named scenario, in addition to the rules the scenario derives from the profile.
func (s *RobustnessTestSuite) AddScenarioFaults(scenario string, fn func(ControllerProfile, *FaultRegistry) []FaultRule) {
	if s.scenarioFaults == nil {
		s.scenarioFaults = map[string][]func(ControllerProfile, *FaultRegistry) []FaultRule{}
	}
	s.scenarioFaults[scenario] = append(s.scenarioFaults[scenario], fn)
}

// AddScenarioVerify registers an extra post-convergence check for the named scenario.
func (s *RobustnessTestSuite) AddScenarioVerify(scenario string, fn func(ControllerProfile, *RobustnessTestFixture)) {
	if s.scenarioVerify == nil {
		s.scenarioVerify = map[string][]func(ControllerProfile, *RobustnessTestFixture){}
	}
	s.scenarioVerify[scenario] = append(s.scenarioVerify[scenario], fn)
}

// SetControllerSetup registers the setup callback for starting the controller.
func (s *RobustnessTestSuite) SetControllerSetup(fn ControllerSetupFn) {
	s.controllerSetup = fn
}

// SetScenarioAction registers the action that triggers reconciliation.
func (s *RobustnessTestSuite) SetScenarioAction(fn ScenarioActionFn) {
	s.scenarioAction = fn
}

// AddSafetyInvariant registers a safety condition checked continuously in the background.
func (s *RobustnessTestSuite) AddSafetyInvariant(name string, fn Invariant) {
	s.continuousInvariants = append(s.continuousInvariants, NamedInvariant{Name: name, Fn: fn})
}

// AddLivenessInvariant registers a target convergence condition. Multiple liveness
// invariants can be registered; all must hold upon convergence.
func (s *RobustnessTestSuite) AddLivenessInvariant(name string, fn Invariant) {
	s.livenessInvariants = append(s.livenessInvariants, NamedInvariant{Name: name, Fn: fn})
}

// SetLivenessTimeout overrides the default 15s convergence timeout ceiling.
func (s *RobustnessTestSuite) SetLivenessTimeout(timeout time.Duration) {
	s.livenessTimeout = timeout
}

// CheckWhenSettled waits for the controller to go write-idle for quietWindow
// before evaluating liveness invariants once (rather than polling until first pass).
func (s *RobustnessTestSuite) CheckWhenSettled(quietWindow time.Duration) {
	s.checkWhenSettled = true
	s.quietWindow = quietWindow
}

// Run executes the trigger action against the controller under every scenario in the matrix.
func (s *RobustnessTestSuite) Run() {
	if s.controllerSetup == nil {
		s.tCtx.Fatal("Setup error: ControllerSetupFn must be set before calling Run()")
	}
	if s.scenarioAction == nil {
		s.tCtx.Fatal("Setup error: ScenarioActionFn must be set before calling Run()")
	}
	if len(s.livenessInvariants) == 0 {
		s.tCtx.Fatal("Setup error: at least one LivenessInvariant must be set before calling Run()")
	}

	for _, scenario := range s.matrix {
		s.tCtx.Run(scenario.Name, func(tCtx ktesting.TContext) {
			tCtx.Logf("=== Robustness scenario %q for controller %q ===", scenario.Name, s.profile.Name)

			// 1. Derive the scenario's faults; skip if none apply to this profile.
			registry := NewFaultRegistry()
			var rules []FaultRule
			if scenario.Faults != nil {
				rules = scenario.Faults(s.profile, registry)
				if len(rules) == 0 {
					tCtx.Skipf("scenario %q has no applicable injection site for controller profile %q", scenario.Name, s.profile.Name)
				}
			}
			for _, extra := range s.scenarioFaults[scenario.Name] {
				rules = append(rules, extra(s.profile, registry)...)
			}

			// 2. Fresh API server and fixture for this cell.
			fixture := newFixture(tCtx, registry)
			defer fixture.TearDown()
			ctx := fixture.Context()

			// 3. Register continuous safety invariants.
			for _, inv := range s.continuousInvariants {
				fixture.AddContinuousInvariant(inv.Name, inv.Fn)
			}

			// 4. Start the controller under test, then arm the scenario's faults.
			s.controllerSetup(tCtx, fixture)
			for _, rule := range rules {
				fixture.InjectFault(rule)
			}

			// 5. Execute the trigger action, then signal completion for phase-gated faults.
			tCtx.Logf("Executing scenario trigger action...")
			if err := s.scenarioAction(ctx, fixture); err != nil {
				tCtx.Fatalf("Scenario trigger action failed: %v", err)
			}
			fixture.Registry().Signal(SignalTriggerComplete)

			// 6. Assert convergence (liveness) while safety invariants run in the background.
			var livenessNames []string
			for _, inv := range s.livenessInvariants {
				livenessNames = append(livenessNames, inv.Name)
			}
			allLiveness := strings.Join(livenessNames, ", ")
			compositeLiveness := func(ctx context.Context, c clientset.Interface) error {
				for _, inv := range s.livenessInvariants {
					if err := inv.Fn(ctx, c); err != nil {
						return fmt.Errorf("liveness invariant %q: %w", inv.Name, err)
					}
				}
				return nil
			}

			if s.checkWhenSettled {
				tCtx.Logf("Waiting for the controller to settle, then checking liveness (%s)...", allLiveness)
				fixture.AssertWhenSettled(allLiveness, compositeLiveness, s.quietWindow, s.livenessTimeout)
			} else {
				tCtx.Logf("Waiting for liveness convergence under faults (%s)...", allLiveness)
				fixture.AssertEventually(allLiveness, compositeLiveness, s.livenessTimeout)
			}

			// 7. A scenario is not over while an injected stall is still in
			// effect: wait it out, then confirm the controller recovered.
			if fixture.WaitForBlockingFaults(s.livenessTimeout) {
				tCtx.Logf("Injected stalls released; re-checking liveness (%s)...", allLiveness)
				if s.checkWhenSettled {
					fixture.AssertWhenSettled(allLiveness, compositeLiveness, s.quietWindow, s.livenessTimeout)
				} else {
					fixture.AssertEventually(allLiveness, compositeLiveness, s.livenessTimeout)
				}
			}

			// 8. Scenario-specific extra assertions, if any.
			if scenario.Verify != nil {
				scenario.Verify(s.profile, fixture)
			}
			for _, verify := range s.scenarioVerify[scenario.Name] {
				verify(s.profile, fixture)
			}

			// 9. Verify fault delivery so mis-targeted rules cannot silently pass.
			fixture.AssertAllFaultsMatched()
			fixture.AssertExpectedFaultsTriggered()

			tCtx.Logf("Scenario %q completed successfully!", scenario.Name)
		})
	}
}
