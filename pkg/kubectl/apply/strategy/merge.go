/*
Copyright 2017 The Kubernetes Authors.

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

package strategy

import "k8s.io/kubernetes/pkg/kubectl/apply"

// Options controls how a merge will be executed
type Options struct {
	// FailOnConflict when true will fail patch creation if the recorded and remote
	// have 2 fields set for the same value that cannot be merged.
	// e.g. primitive values, list values with replace strategy, and map values with do
	// strategy
	FailOnConflict bool
}

// Create returns a new apply.Visitor for merging multiple objects together
func Create(options Options) apply.Strategy {
	return createDelegatingStrategy(options)
}
