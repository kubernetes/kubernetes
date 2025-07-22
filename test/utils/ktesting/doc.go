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

// Package ktesting is a wrapper around k8s.io/klog/v2/ktesting. In contrast
// to the klog package, this one is opinionated and tailored towards testing
// Kubernetes.
//
// Importing it
// - adds the -v command line flag
// - enables better dumping of complex datatypes
// - sets the default verbosity to 5 (can be changed with [SetDefaultVerbosity])
//
// It also adds additional APIs and types for unit and integration tests
// which are too experimental for klog and/or are unrelated to logging.
// The ktesting package itself takes care of managing a test context
// with deadlines, timeouts, cancellation, and some common attributes
// as first-class members of the API. Sub-packages have additional APIs
// for propagating values via the context, implemented via [WithValue].
package ktesting
