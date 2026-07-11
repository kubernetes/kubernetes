/*
Copyright 2015 The Kubernetes Authors.

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

// Package testing provides a fake Kubernetes client suitable for use in unit
// tests. The fake client simulates interactions with a kube-apiserver without
// requiring one to be running, making tests fast and self-contained.
//
// # Scope and Limitations
//
// This fake client is intentionally simplified. It does not, and will never,
// fully replicate the behavior of a real kube-apiserver. Many server-side
// behaviors such as field defaulting, validation, status management,
// strategic merge patch, server-side apply semantics, and advanced
// field selectors are not supported by this client and there are no plans
// to add them.
//
// This is by design. Maintaining a high-fidelity mock of the entire
// kube-apiserver API surface would introduce significant complexity that is
// difficult to justify for a test utility.
//
// # When to Use This Package
//
// The fake client works well for unit tests that need to verify how your code
// interacts with the Kubernetes API at a structural level, for example:
//
//   - Verifying that the correct API calls are made.
//   - Supplying canned responses to drive specific code paths.
//
// # When Not to Use This Package
//
// If your tests depend on the kube-apiserver behaving correctly (e.g.,
// enforcing validation, persisting resources accurately, handling apply
// semantics, or producing realistic watch events), you should write
// integration tests against a real kube-apiserver instead.
//
// # Contributing
//
// Issues requesting that the fake client more closely match kube-apiserver
// behavior should be limiting to bugs in how the fake behaves for unit test
// scenarios it is clearly intended to support. Pull requests that improve the fake
// client will only be accepted when they meet all of the following criteria:
//
//   - The change makes the fake client easier to use for common unit testing
//     patterns.
//   - The change does not introduce significant complexity to the fake client.
//   - The use cases motivating the change are clearly better served by a fake
//     client than by integration tests against a real kube-apiserver.
//
// We hold a high bar for these changes. If the test scenarios in question can
// be reasonably addressed through integration testing, we will prefer that
// path over expanding the fake client.
//
// We understand this stance may be inconvenient, and we appreciate your
// understanding. Our goal is to keep this package simple, maintainable, and
// honest about what it provides so that it remains a reliable tool for the
// cases it is designed to handle.
package testing
