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

	clientset "k8s.io/client-go/kubernetes"
)

// ControllerSetupFn initializes and runs the controller under test using the fixture's wrapped clients.
type ControllerSetupFn func(fixture *RobustnessTestFixture)

// ScenarioActionFn executes the action that triggers reconciliation.
type ScenarioActionFn func(ctx context.Context, fixture *RobustnessTestFixture) error

// RobustnessTestSuite runs a controller's trigger action and invariants across
// every scenario of a chaos matrix.
type RobustnessTestSuite struct {
	t       *testing.T
	profile ControllerProfile
	matrix  []ChaosScenario

	controllerSetup      ControllerSetupFn
	scenarioAction       ScenarioActionFn
	continuousInvariants []NamedInvariant
	livenessInvariants   []NamedInvariant
	livenessTimeout      time.Duration

	checkWhenSettled bool
	quietWindow      time.Duration
}

// NewTestSuite creates a chaos-matrix runner for the controller described by
// profile, preloaded with the standard scenario matrix.
func NewTestSuite(t *testing.T, profile ControllerProfile) *RobustnessTestSuite {
	if err := profile.validate(); err != nil {
		t.Fatalf("invalid ControllerProfile: %v", err)
	}
	return &RobustnessTestSuite{
		t:               t,
		profile:         profile,
		matrix:          StandardChaosMatrix(),
		livenessTimeout: 30 * time.Second,
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

// SetLivenessTimeout overrides the default 30s convergence timeout ceiling.
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
		s.t.Fatal("Setup error: ControllerSetupFn must be set before calling Run()")
	}
	if s.scenarioAction == nil {
		s.t.Fatal("Setup error: ScenarioActionFn must be set before calling Run()")
	}
	if len(s.livenessInvariants) == 0 {
		s.t.Fatal("Setup error: at least one LivenessInvariant must be set before calling Run()")
	}

	for _, scenario := range s.matrix {
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

			// 5. Execute the trigger action, then signal completion for phase-gated faults.
			t.Logf("Executing scenario trigger action...")
			if err := s.scenarioAction(ctx, fixture); err != nil {
				t.Fatalf("Scenario trigger action failed: %v", err)
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
				t.Logf("Waiting for the controller to settle, then checking liveness (%s)...", allLiveness)
				fixture.AssertWhenSettled(allLiveness, compositeLiveness, s.quietWindow, s.livenessTimeout)
			} else {
				t.Logf("Waiting for liveness convergence under faults (%s)...", allLiveness)
				fixture.AssertEventually(allLiveness, compositeLiveness, s.livenessTimeout)
			}

			// 7. Scenario-specific extra assertions, if any.
			if scenario.Verify != nil {
				scenario.Verify(s.profile, fixture)
			}

			// 8. Verify fault delivery so mis-targeted rules cannot silently pass.
			fixture.AssertAllFaultsMatched()
			fixture.AssertExpectedFaultsTriggered()

			t.Logf("Scenario %q completed successfully!", scenario.Name)
		})
	}
}
