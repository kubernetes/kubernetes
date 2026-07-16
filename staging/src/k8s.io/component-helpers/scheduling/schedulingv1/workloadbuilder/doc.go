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

// Package workloadbuilder is the shared translation library from KEP-6089. It
// turns a controller's scheduling intent into the scheduler-facing
// scheduling.k8s.io Workload, handling defaulting, validation, and
// PodGroupTemplate compilation so controllers don't each reimplement it.
//
// A controller maps its API into a SchedulingConfig (MapPodGroupConfig),
// assembles a WorkloadItem tree, calls Build() to compile the Workload, then
// NewPodGroup() to materialize its runtime PodGroup.
//
// For validating a controller's own API, use ValidateSchedulingPolicy/Update and
// ValidateDisruptionMode/Update. They combine the controller's capability
// allow-list with the generated declarative validation (DV) for the
// building-block types, so out-of-tree controllers enforce the same structural
// constraints the kube-apiserver applies to built-in types. Validate() runs the
// full pre-Build compile check over an already-resolved tree.
//
// Build and Validate return a plain error describing why compilation failed,
// using paths relative to the library's internal IR.
package workloadbuilder
