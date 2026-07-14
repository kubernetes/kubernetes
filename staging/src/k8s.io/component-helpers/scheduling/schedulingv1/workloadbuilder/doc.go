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
// A controller maps its API into a SchedulingConfig (MapPodGroupConfig) and
// assembles a WorkloadItem tree, then drives everything through a Builder:
//
//	builder := workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
//		Name:                   "trainer-wl",
//		Namespace:              "ml",
//		Owner:                  ownerRef,
//		AllowedPolicies:        []workloadbuilder.SchedulingPolicyOption{workloadbuilder.BasicPolicy, workloadbuilder.GangPolicy},
//		AllowedDisruptionModes: []workloadbuilder.DisruptionModeOption{workloadbuilder.SingleMode, workloadbuilder.AllMode},
//	})
//	if errs := builder.Validate(context.Background(), operation.Operation{Type: operation.Create}, nil, fldPath); len(errs) > 0 { /* reject */ }
//	workload, err := builder.BuildWorkload()
//	podGroup, err := builder.NewPodGroup("trainer-pg", "trainer-pgt-0")
//
// Validate runs the controller-policy checks declarative validation cannot
// express: the resolved policy and disruption mode must be in the Builder's
// allow-lists, and the Basic policy cannot be combined with All disruption.
// In-tree controllers also get structural building-block checks from generated
// declarative validation at the apiserver; out-of-tree controllers must call
// those Validate_* functions directly before builder.Validate.
//
// BuildWorkload compiles and caches the Workload so multiple PodGroups can be
// materialized from one compiled result. A controller managing a PodGroup for a
// Workload it did not compile calls SetExistingWorkload instead: the Builder
// then materializes PodGroups from that Workload and refuses BuildWorkload so
// the supplied object is never recompiled over.
//
// Validation splits into two layers:
//
//   - Declarative validation (DV): structural constraints on the scheduling.k8s.io
//     building blocks (union cardinality, minCount bounds, formats). Generated
//     validators live in-package on k8s.io/api/scheduling/v1alpha3 as Validate_*
//     functions.
//
//   - Complex validation: controller allow-lists and cross-field rules DV cannot
//     express (e.g. Basic policy with All disruption). Builder.Validate runs
//     these against a WorkloadItem tree.
//
// In-tree controllers get DV automatically from the apiserver; hand-written
// admission validation calls only builder.Validate for the complex checks.
//
// Out-of-tree controllers must run both. Call the generated DV validators on
// the versioned building-block types directly, then builder.Validate on the
// mapped WorkloadItem:
//
//	import (
//	    "context"
//	    schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
//	    "k8s.io/apimachinery/pkg/api/operation"
//	)
//	op := operation.Operation{Type: operation.Create}
//	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, fldPath.Child("policy"), policy, nil)...)
//	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, fldPath.Child("disruptionMode"), mode, nil)...)
//	builder := workloadbuilder.NewBuilder(item, opts)
//	allErrs = append(allErrs, builder.Validate(context.Background(), operation.Operation{Type: operation.Create}, nil, fldPath)...)
//
// See ExampleoutOfTreeControllerValidation.
package workloadbuilder
