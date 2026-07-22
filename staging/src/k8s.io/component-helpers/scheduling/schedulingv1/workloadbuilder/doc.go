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

// Package workloadbuilder is a shared translation library that turns
// a controller's scheduling intent into the scheduler-facing Workload,
// handling defaulting, validation, and PodGroupTemplate compilation so
// controllers don't each reimplement it.
//
// A controller records its API's scheduling building blocks (and the field
// paths where they live) in a WorkloadItem's Input, layers them over a
// DefaultConfig, assembles the WorkloadItem tree, then drives everything
// through a Builder:
//
//	builder := workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
//		Name:                   "trainer-wl",
//		Namespace:              "ml",
//		Owner:                  ownerRef,
//		AllowedPolicies:        []workloadbuilder.SchedulingPolicyOption{workloadbuilder.BasicPolicy, workloadbuilder.GangPolicy},
//		AllowedDisruptionModes: []workloadbuilder.DisruptionModeOption{workloadbuilder.SingleMode, workloadbuilder.AllMode},
//	})
//	if errs := builder.Validate(ctx, workloadbuilder.ValidationInput{}); len(errs) > 0 { /* reject */ }
//	workload, err := builder.BuildWorkload()
//	podGroup, err := builder.NewPodGroup("trainer-pg", "trainer-pgt-0")
//
// BuildWorkload compiles and caches the Workload so multiple PodGroups can be
// materialized from one compiled result. A controller managing a PodGroup for a
// Workload it did not compile uses NewBuilderFromExistingWorkload instead, the
// Builder then materializes PodGroups from that Workload and refuses
// BuildWorkload so the supplied object is never recompiled over.
//
// # Validation
//
// Builder.Validate reports errors in two layers against the WorkloadItem tree:
//
//   - Declarative validation (DV): structural constraints on the
//     building blocks recorded in each Input. Validate runs the
//     generated Validate_* validators against those blocks and
//     reports at the field paths the controller recorded in the Input's
//     PathElements.
//
//   - Complex validation: the controller allow-lists (AllowedPolicies,
//     AllowedDisruptionModes) and cross-field rules DV cannot express, such as
//     the Basic policy not being combinable with All disruption.
//
// Out-of-tree controllers leave DV enabled so one Builder.Validate call covers
// both layers. They record each building block with the field path where it
// lives in the controller's API, then call Validate once:
//
//	item := &workloadbuilder.WorkloadItem{
//		Name: "worker",
//		Path: fldPath,
//		Input: workloadbuilder.WorkloadInput{
//			Policy:         workloadbuilder.PolicyInput{PodGroupData: policy, PathElements: []string{"schedulingPolicy"}},
//			DisruptionMode: workloadbuilder.DisruptionModeInput{PodGroupData: mode, PathElements: []string{"disruptionMode"}},
//		},
//	}
//	builder := workloadbuilder.NewBuilder(item, opts)
//	allErrs = append(allErrs, builder.Validate(ctx, workloadbuilder.ValidationInput{})...)
//
// The zero ValidationInput validates a create. On an update the controller sets
// OldRoot so DV runs the update-time checks (i.e. immutability) against the
// previously persisted item; the operation is inferred from OldRoot being
// non-nil:
//
//	allErrs = append(allErrs, builder.Validate(ctx, workloadbuilder.ValidationInput{
//		OldRoot: oldItem,
//	})...)
//
// See ExampleOutOfTreeControllerValidation for a runnable version. An in-tree
// controller whose apiserver already runs DV on the embedded building blocks
// sets BuildOptions.DisableDeclarativeValidation to skip the first layer and run
// only the complex checks; it passes the zero ValidationInput regardless of the
// operation because OldRoot is only consulted by DV. See ExampleJobControllerE2E.
package workloadbuilder
