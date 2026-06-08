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

package examples

import (
	"context"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
)

// ExampleOutOfTreeControllerValidation shows how an out-of-tree controller
// validates its scheduling API at admission. It records each building block and
// the field path where it lives in the WorkloadItem's Input, then calls
// Builder.Validate, which runs declarative validation on the building blocks
// followed by the complex checks that DV cannot express. In-tree controllers
// get DV from the apiserver, so they instead set
// BuildOptions.DisableDeclarativeValidation.
func ExampleOutOfTreeControllerValidation() {
	ctx := context.Background()
	schedulingPath := field.NewPath("spec", "scheduling")

	policy := &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
	}
	mode := &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
		All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
	}

	item := &workloadbuilder.WorkloadItem{
		Name: "worker",
		DefaultConfig: &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
		},
		Input: workloadbuilder.WorkloadInput{
			Policy:         workloadbuilder.PolicyInput{PodGroupData: policy, PathElements: []string{"schedulingPolicy"}},
			DisruptionMode: workloadbuilder.DisruptionModeInput{PodGroupData: mode, PathElements: []string{"disruptionMode"}},
		},
	}
	// The zero ValidationInput validates a create. On an update, the controller
	// passes ValidationInput{OldRoot: oldItem} so DV runs the update-time checks
	// (i.e. immutability) against the previously persisted item; the operation is
	// inferred from OldRoot being non-nil.
	allErrs := workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
		Owner:                  &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "worker", UID: "job-uid"},
		AllowedPolicies:        []workloadbuilder.SchedulingPolicyOption{workloadbuilder.BasicPolicy, workloadbuilder.GangPolicy},
		AllowedDisruptionModes: []workloadbuilder.DisruptionModeOption{workloadbuilder.SingleMode, workloadbuilder.AllMode},
	}).Validate(ctx, schedulingPath, workloadbuilder.ValidationInput{})

	if len(allErrs) > 0 {
		fmt.Printf("rejected: %v\n", allErrs.ToAggregate())
		return
	}
	fmt.Println("accepted")

	// Output:
	// rejected: spec.scheduling.disruptionMode: Invalid value: "": the disruptionMode `all` is not supported with the Basic scheduling policy
}

// ExampleJobControllerE2E shows the end-to-end flow a controller uses to turn its
// user-facing scheduling API into a scheduler-facing Workload and a runtime
// PodGroup. It mirrors what the built-in Job controller does for a single Job:
// map spec.scheduling, layer it over a controller default, default the gang
// minCount to the Job's parallelism, compile the Workload, then materialize the
// PodGroup from the compiled template.
func ExampleJobControllerE2E() {
	// The Job's parallelism, used to default an unset gang minCount.
	parallelism := int32(4)

	// 1. Record the user's spec.scheduling (here: gang with no minCount, pinned
	//    to a single node per topology) as the versioned building blocks paired
	//    with the field paths where the controller embeds them. A nil sub-field
	//    is left nil so it can fall back to the controller default field-by-field.
	userInput := workloadbuilder.WorkloadInput{
		Policy: workloadbuilder.PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}, // minCount defaulted below
		}, PathElements: []string{"schedulingPolicy"}},
		Constraints: workloadbuilder.ConstraintsInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
			Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "kubernetes.io/hostname"}},
		}, PathElements: []string{"schedulingConstraints"}},
		// DisruptionMode and ResourceClaims left unset: use the defaults.
	}

	// 2. Assemble the single-node workload tree: a controller default (Basic),
	//    the user's intent layered on top, and a callback that defaults the gang
	//    minCount to the Job's parallelism when the user left it unset. The
	//    callback mutates only the resolved config, never the caller's inputs.
	item := &workloadbuilder.WorkloadItem{
		Name:          "trainer-pgt-0",
		DefaultConfig: &workloadbuilder.SchedulingConfig{Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}}},
		Input:         userInput,
		Callbacks: []workloadbuilder.SchedulingConfigFunc{
			func(cfg *workloadbuilder.SchedulingConfig) {
				if g := cfg.Policy.Gang; g != nil && g.MinCount == nil {
					minCount := parallelism
					g.MinCount = &minCount
				}
			},
		},
	}

	// 3. Construct the builder with the Workload's identity, the controller
	//    ownerRef (which becomes spec.controllerRef, used to discover the
	//    Workload and drive garbage collection), and the controller's scheduling
	//    allow-lists.
	owner := &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "trainer", UID: "job-uid"}
	builder := workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
		Name:                   "trainer-wl",
		Namespace:              "ml",
		Owner:                  owner,
		AllowedPolicies:        []workloadbuilder.SchedulingPolicyOption{workloadbuilder.BasicPolicy, workloadbuilder.GangPolicy},
		AllowedDisruptionModes: []workloadbuilder.DisruptionModeOption{workloadbuilder.SingleMode, workloadbuilder.AllMode},
		// The Job apiserver already runs declarative validation on the building
		// blocks, so the in-tree controller opts out of running it again here and
		// Validate performs only the complex controller-policy checks.
		DisableDeclarativeValidation: true,
	})

	// 4. Run the complex controller-policy checks. Structural building-block
	// validation already ran at the apiserver, so it is disabled above.
	if errs := builder.Validate(context.Background(), field.NewPath("spec", "scheduling"), workloadbuilder.ValidationInput{}); len(errs) > 0 {
		fmt.Printf("validation failed: %v\n", errs.ToAggregate())
		return
	}

	// 5. Compile the Workload. The result is cached so PodGroups can be
	//    materialized without recompiling.
	workload, err := builder.BuildWorkload()
	if err != nil {
		fmt.Printf("build failed: %v\n", err)
		return
	}

	// 6. Materialize the runtime PodGroup from the compiled template. It is owned
	//    by the builder's configured owner.
	podGroup, err := builder.NewPodGroup("trainer-pg", item.Name)
	if err != nil {
		fmt.Printf("podgroup failed: %v\n", err)
		return
	}

	tmpl := workload.Spec.PodGroupTemplates[0]
	fmt.Printf("Workload %s/%s controllerRef=%s/%s/%s\n",
		workload.Namespace, workload.Name,
		workload.Spec.ControllerRef.APIGroup, workload.Spec.ControllerRef.Kind, workload.Spec.ControllerRef.Name)
	fmt.Printf("  template %q: gang minCount=%d, topology=%s\n",
		tmpl.Name, tmpl.SchedulingPolicy.Gang.MinCount, tmpl.SchedulingConstraints.Topology[0].Key)
	fmt.Printf("PodGroup %s/%s -> workload=%s template=%s gang minCount=%d\n",
		podGroup.Namespace, podGroup.Name,
		podGroup.Spec.WorkloadRef.WorkloadName, podGroup.Spec.WorkloadRef.TemplateName,
		podGroup.Spec.SchedulingPolicy.Gang.MinCount)

	// Output:
	// Workload ml/trainer-wl controllerRef=batch/Job/trainer
	//   template "trainer-pgt-0": gang minCount=4, topology=kubernetes.io/hostname
	// PodGroup ml/trainer-pg -> workload=trainer-wl template=trainer-pgt-0 gang minCount=4
}

// ExampleDelegatedPodGroupFromExistingWorkload shows how a controller
// materializes a runtime PodGroup from a Workload it did not compile. A parent
// controller (or a hand-authored template) owns and already persisted the
// Workload, so it is the source of truth: NewBuilderFromExistingWorkload takes
// that Workload directly and the child controller supplies no scheduling inputs
// of its own. Validate is a no-op (the persisted object already passed apiserver
// validation) and BuildWorkload is refused so the object is never recompiled
// over. This mirrors the Job controller's delegated PodGroup mode.
func ExampleDelegatedPodGroupFromExistingWorkload() {
	// A parent controller already compiled and persisted this Workload.
	parentWorkload := &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "trainer-wl", Namespace: "ml", UID: "wl-uid"},
		Spec: schedulingv1alpha3.WorkloadSpec{
			PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{{
				Name:              "trainer-pgt-0",
				PriorityClassName: "high-priority",
				SchedulingPolicy:  schedulingv1alpha3.PodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4}},
			}},
		},
	}

	// The child controller owns only the PodGroup, so it passes its own ownerRef.
	owner := &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "trainer-worker", UID: "job-uid"}
	builder := workloadbuilder.NewBuilderFromExistingWorkload(parentWorkload, workloadbuilder.BuildOptions{Owner: owner})

	// Validate is a no-op for a builder created from an existing Workload.
	if errs := builder.Validate(context.Background(), field.NewPath("spec", "scheduling"), workloadbuilder.ValidationInput{}); len(errs) > 0 {
		fmt.Printf("unexpected validation errors: %v\n", errs.ToAggregate())
		return
	}

	// BuildWorkload is refused so the parent-authored Workload is never recompiled.
	if _, err := builder.BuildWorkload(); err != nil {
		fmt.Println("BuildWorkload refused for existing Workload")
	}

	// Only NewPodGroup is meaningful: it copies the persisted template's fields
	// (here priorityClassName and the gang minCount) into the runtime PodGroup.
	podGroup, err := builder.NewPodGroup("trainer-pg", "trainer-pgt-0")
	if err != nil {
		fmt.Printf("podgroup failed: %v\n", err)
		return
	}

	fmt.Printf("PodGroup %s/%s -> workload=%s template=%s priorityClass=%s gang minCount=%d ownedBy=%s\n",
		podGroup.Namespace, podGroup.Name,
		podGroup.Spec.WorkloadRef.WorkloadName, podGroup.Spec.WorkloadRef.TemplateName,
		podGroup.Spec.PriorityClassName,
		podGroup.Spec.SchedulingPolicy.Gang.MinCount,
		podGroup.OwnerReferences[0].Name)

	// Output:
	// BuildWorkload refused for existing Workload
	// PodGroup ml/trainer-pg -> workload=trainer-wl template=trainer-pgt-0 priorityClass=high-priority gang minCount=4 ownedBy=trainer-worker
}
