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
// loops: a controller author declares what their controller does and what must
// hold true, and the framework runs the controller under a matrix of injected
// failures, verifying the declared invariants throughout.
//
// The pieces, in the order a test author meets them:
//
//   - ControllerProfile (profile.go) declares what the controller does through
//     the wrapped client: the root resource it reconciles, which writes it
//     performs, whether it creates child objects, and whether it uses
//     ControllerExpectations. Scenarios derive their faults from the profile,
//     which is what makes the matrix reusable for any controller.
//
//   - RobustnessTestSuite (suite.go) orchestrates the matrix: for each scenario
//     it builds a fresh fixture, starts the controller, injects the scenario's
//     faults, runs the trigger action, and checks the invariants.
//
//   - Invariants (invariants.go) come in two flavors: safety invariants are
//     checked continuously in the background ("never more than one Pod per
//     node"), and the liveness invariant is the convergence target ("status
//     eventually reports 1 scheduled"). Both run against the un-wrapped admin
//     client. ObjectSatisfies and CountAtMost build common shapes.
//
//   - ChaosScenario (scenarios.go) is one cell of the matrix: a self-contained
//     fault set derived from the profile. StandardChaosMatrix covers write
//     conflicts, cache sync lag, flaky child creates, combined chaos, and
//     expectations timeouts; AddScenario appends custom cells.
//
//   - FaultRegistry (registry.go) is the single source of truth for active
//     faults. A FaultRule pairs a structured FaultMatch (ClientMatch,
//     CacheMatch, ClockMatch, QueueMatch) with a FaultCondition (when to fire)
//     and a FaultAction (what to do), validated against the match's injection
//     domain at registration.
//
//   - Injection sites wrap the controller's dependencies and consult the
//     registry: the REST transport (transport.go), informer caches
//     (indexer.go, informers.go), the expectations clock (clock.go), and work
//     queues (queue.go). The fixture (fixture.go) wires them together around an
//     integration apiserver.
//
// Two guards keep a green run meaningful: every non-optional fault must have
// matched a real injection site (AssertAllFaultsMatched), and every fault
// declared ExpectTriggered must have actually fired
// (AssertExpectedFaultsTriggered). A fault that silently injects nothing is a
// test failure, not a pass.
//
// See test/integration/robustness for complete examples (lease renewal and the
// DaemonSet controller).
package robustness
