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
	"testing"
	"time"
)

// ControllerSetupFn initializes and runs the controller under test, using the fixture's wrapped clients.
type ControllerSetupFn func(fixture *RobustnessTestFixture)

// ScenarioActionFn executes the action that triggers reconciliation, using direct or wrapped clients from the fixture.
type ScenarioActionFn func(ctx context.Context, fixture *RobustnessTestFixture) error

// RobustnessTestSuite runs a controller's trigger action and invariants under
// every scenario of a chaos matrix. The controller is described declaratively
// by a ControllerProfile; scenarios derive their faults from it, so the same
// matrix applies to any controller without modification.
type RobustnessTestSuite struct {
	t       *testing.T
	profile ControllerProfile
	matrix  []ChaosScenario

	controllerSetup      ControllerSetupFn
	scenarioAction       ScenarioActionFn
	continuousInvariants []NamedInvariant
	livenessInvariant    NamedInvariant
	livenessTimeout      time.Duration

	// Settled-state configuration. When checkWhenSettled is set, the liveness
	// invariant is checked once after the controller settles (write-idle for
	// quietWindow) rather than polled until it first holds. This gives
	// deterministic "end of reconciliation" semantics for convergent controllers.
	checkWhenSettled bool
	quietWindow      time.Duration

	scenarioCustomizers map[string]func(fixture *RobustnessTestFixture)
}

// NewTestSuite creates a chaos-matrix runner for the controller described by
// profile, preloaded with the standard scenario matrix.
func NewTestSuite(t *testing.T, profile ControllerProfile) *RobustnessTestSuite {
	if err := profile.validate(); err != nil {
		t.Fatalf("invalid ControllerProfile: %v", err)
	}
	return &RobustnessTestSuite{
		t:       t,
		profile: profile,
		matrix:  StandardChaosMatrix(),
		// Generous default ceiling: PollUntilContextTimeout returns as soon as the
		// invariant holds, so a higher timeout only affects how long we wait before
		// declaring failure — it does not slow the passing path. Integration
		// convergence under load (fresh apiserver per subtest, shared etcd) can
		// exceed 10s, which made tight ceilings flaky.
		livenessTimeout:     30 * time.Second,
		scenarioCustomizers: make(map[string]func(fixture *RobustnessTestFixture)),
	}
}

// Profile returns the declared controller profile.
func (s *RobustnessTestSuite) Profile() ControllerProfile {
	return s.profile
}

// AddScenario appends a custom scenario to the matrix, after the standard ones.
func (s *RobustnessTestSuite) AddScenario(scenario ChaosScenario) {
	s.matrix = append(s.matrix, scenario)
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

// SetLivenessInvariant registers the target convergence state of the system.
func (s *RobustnessTestSuite) SetLivenessInvariant(name string, fn Invariant, timeout time.Duration) {
	s.livenessInvariant = NamedInvariant{Name: name, Fn: fn}
	s.livenessTimeout = timeout
}

// CheckWhenSettled opts into deterministic end-of-reconciliation checking: instead
// of polling the liveness invariant until it first holds, the suite waits for the
// controller to settle (no mutating requests for quietWindow) and then evaluates
// the invariant exactly once. Use this for convergent controllers; leave it unset
// for controllers that write continuously (e.g. periodic lease renewal), which
// should keep the polling AssertEventually behavior.
func (s *RobustnessTestSuite) CheckWhenSettled(quietWindow time.Duration) {
	s.checkWhenSettled = true
	s.quietWindow = quietWindow
}

// ConfigureScenario registers a custom fault-injection decorator for a specific
// scenario in the matrix, run after the scenario's own faults are injected.
func (s *RobustnessTestSuite) ConfigureScenario(scenarioName string, fn func(fixture *RobustnessTestFixture)) {
	s.scenarioCustomizers[scenarioName] = fn
}

// Run executes the trigger action against the controller under every scenario in the matrix.
func (s *RobustnessTestSuite) Run() {
	if s.controllerSetup == nil {
		s.t.Fatal("Setup error: ControllerSetupFn must be set before calling Run()")
	}
	if s.scenarioAction == nil {
		s.t.Fatal("Setup error: ScenarioActionFn must be set before calling Run()")
	}
	if s.livenessInvariant.Fn == nil {
		s.t.Fatal("Setup error: LivenessInvariant must be set before calling Run()")
	}

	for _, scenario := range s.matrix {
		scenario := scenario // Capture variable
		s.t.Run(scenario.Name, func(t *testing.T) {
			t.Logf("=== Robustness scenario %q for controller %q ===", scenario.Name, s.profile.Name)

			// 1. Fresh test environment with a namespace-safe prefix (max 20 chars).
			prefix := strings.ToLower(fmt.Sprintf("r-%s", scenario.Name))
			if len(prefix) > 20 {
				prefix = prefix[:20]
			}
			fixture := NewFixture(t, prefix)
			defer fixture.TearDown()
			ctx := fixture.Context()

			// 2. Register continuous safety invariants.
			for _, inv := range s.continuousInvariants {
				fixture.AddContinuousInvariant(inv.Name, inv.Fn)
			}

			// 3. Start the controller under test.
			s.controllerSetup(fixture)

			// 4. Inject the scenario's faults, derived from the controller profile.
			if scenario.Faults != nil {
				for _, rule := range scenario.Faults(s.profile, fixture.Registry()) {
					fixture.InjectFault(rule)
				}
			}

			// 5. Run user-defined scenario customization if registered.
			if customizer, ok := s.scenarioCustomizers[scenario.Name]; ok {
				customizer(fixture)
			}

			// 6. Execute the trigger action, then signal completion so
			// phase-gated faults (TriggerAfterSignal) can begin firing.
			t.Logf("Executing scenario trigger action...")
			if err := s.scenarioAction(ctx, fixture); err != nil {
				t.Fatalf("Scenario trigger action failed: %v", err)
			}
			fixture.Registry().Signal(SignalTriggerComplete)

			// 7. Assert convergence (liveness), while safety invariants check in
			// the background. When settled-state checking is enabled, wait for
			// the controller to settle and check the final state once; otherwise
			// poll until the invariant first holds.
			if s.checkWhenSettled {
				t.Logf("Waiting for the controller to settle, then checking liveness (%s)...", s.livenessInvariant.Name)
				fixture.AssertWhenSettled(s.livenessInvariant.Name, s.livenessInvariant.Fn, s.quietWindow, s.livenessTimeout)
			} else {
				t.Logf("Waiting for liveness convergence under faults (%s)...", s.livenessInvariant.Name)
				fixture.AssertEventually(s.livenessInvariant.Name, s.livenessInvariant.Fn, s.livenessTimeout)
			}

			// 8. Scenario-specific extra assertions, if any.
			if scenario.Verify != nil {
				scenario.Verify(s.profile, fixture)
			}

			// 9. Guard against silent no-op faults: every non-optional fault must
			// have matched a real injection site, and every fault declared
			// ExpectTriggered must have actually fired. A green result without
			// these checks would be meaningless.
			fixture.AssertAllFaultsMatched()
			fixture.AssertExpectedFaultsTriggered()

			t.Logf("Scenario %q completed successfully!", scenario.Name)
		})
	}
}
