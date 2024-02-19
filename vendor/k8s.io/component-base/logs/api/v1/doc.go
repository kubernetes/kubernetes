/*
Copyright 2022 The Kubernetes Authors.

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

// +k8s:deepcopy-gen=package

// Package v1 contains the configuration API for logging.
//
// The intention is to only have a single version of this API, potentially with
// new fields added over time in a backwards-compatible manner. Fields for
// alpha or beta features are allowed as long as they are defined so that not
// changing the defaults leaves those features disabled.
//
// The "v1" package name is just a reminder that API compatibility rules apply,
// not an indication of the stability of all features covered by it.

// The LoggingAlphaOptions and LoggingBetaOptions feature gates control whether
// these unstable features can get enabled. This can be used to ensure that
// command invocations do not accidentally rely on unstable features.
package v1 // import "k8s.io/component-base/logs/api/v1"
