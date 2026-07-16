/*
Copyright 2023 The Kubernetes Authors.

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

// Package ktesting exists in different variants, each adding more
// functionality on top of the previous one:
//   - [k8s.io/klog/v2/ktesting]: contextual logging, caller determines
//     all aspects
//   - [k8s.io/kubernetes/test/utils/ktesting]: abstraction around
//     Go testing and Ginkgo
//   - [k8s.io/kubernetes/test/utils/client-go/ktesting]: tailored towards
//     testing Kubernetes
//
// Consumers can switch from a simpler variant to a more complex one simply
// changing the import path. The more complex ones have additional package
// dependencies.
//
// This variant is opinionated. Importing it:
// - adds the -v command line flag
// - enables better dumping of complex datatypes
// - sets the default verbosity to 5 (can be changed with [SetDefaultVerbosity])
//
// It also adds additional APIs and types for unit and integration tests
// which are too experimental for klog and/or are unrelated to logging.
// The ktesting package itself takes care of managing a test context
// with deadlines, timeouts, cancellation, and some common attributes
// as first-class members of the API.
package ktesting
