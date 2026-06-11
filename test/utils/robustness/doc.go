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

// Package robustness is a fault-injection harness for controller reconciliation
// loops. A test author declares what their controller does (ControllerProfile)
// and what must hold true (safety and liveness Invariants); the suite runs the
// controller across a matrix of injected faults and verifies the invariants.
//
// Core components:
//
//   - ControllerProfile (profile.go): declares the root resource, writes performed,
//     child resources, and whether ControllerExpectations are used. Scenarios
//     derive their fault rules from the profile so the matrix is reusable across
//     controllers.
//   - RobustnessTestSuite (suite.go): runs each ChaosScenario against a fresh
//     RobustnessTestFixture (fixture.go), executing the trigger action and
//     checking continuous safety invariants and final liveness convergence.
//   - FaultRegistry (registry.go): matches injection sites (ClientMatch,
//     CacheMatch, ClockMatch, QueueMatch) against trigger conditions and
//     domain-validated fault actions.
//
// To prevent silent no-op passes, every non-optional fault rule must match an
// injection site (AssertAllFaultsMatched) and every ExpectTriggered rule must
// fire at least once (AssertExpectedFaultsTriggered).
//
// See test/integration/daemonset and test/integration/node for examples.
package robustness
