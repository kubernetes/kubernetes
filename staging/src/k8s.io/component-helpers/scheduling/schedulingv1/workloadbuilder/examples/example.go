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
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
)

// ExampleoutOfTreeControllerValidation shows how an out-of-tree controller
// validates its scheduling API at admission. It runs generated declarative
// validation on the versioned building blocks, then builder.Validate for the
// complex checks (allow-lists and Basic+All) that DV cannot express. In-tree
// controllers get DV from the apiserver and only need the builder.Validate step.
func ExampleoutOfTreeControllerValidation() {
	ctx := context.Background()
	op := operation.Operation{Type: operation.Create}
	schedulingPath := field.NewPath("spec", "scheduling")

	policy := &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
	}
	mode := &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
		All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
	}

	var allErrs field.ErrorList
	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupSchedulingPolicy(ctx, op, schedulingPath.Child("policy"), policy, nil)...)
	allErrs = append(allErrs, schedulingv1alpha3.Validate_WorkloadPodGroupDisruptionMode(ctx, op, schedulingPath.Child("disruptionMode"), mode, nil)...)

	item := &workloadbuilder.WorkloadItem{
		Name: "worker",
		DefaultConfig: &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
		},
		UserConfig: workloadbuilder.MapPodGroupConfig(workloadbuilder.WorkloadPodGroupConfig{
			Policy:         policy,
			DisruptionMode: mode,
		}),
	}
	allErrs = append(allErrs, workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
		Owner:                  &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "worker", UID: "job-uid"},
		AllowedPolicies:        []workloadbuilder.SchedulingPolicyOption{workloadbuilder.BasicPolicy, workloadbuilder.GangPolicy},
		AllowedDisruptionModes: []workloadbuilder.DisruptionModeOption{workloadbuilder.SingleMode, workloadbuilder.AllMode},
	}).Validate(schedulingPath)...)

	if len(allErrs) > 0 {
		fmt.Printf("rejected: %v\n", allErrs.ToAggregate())
		return
	}
	fmt.Println("accepted")

	// Output:
	// rejected: spec.scheduling.disruptionMode.all: Invalid value: "": the disruptionMode `all` is not supported with the Basic scheduling policy
}

// ExamplejobControllerE2E shows the end-to-end flow a controller uses to turn its
// user-facing scheduling API into a scheduler-facing Workload and a runtime
// PodGroup. It mirrors what the built-in Job controller does for a single Job:
// map spec.scheduling, layer it over a controller default, default the gang
// minCount to the Job's parallelism, compile the Workload, then materialize the
// PodGroup from the compiled template.
func ExamplejobControllerE2E() {
	// The Job's parallelism, used to default an unset gang minCount.
	parallelism := int32(4)

	// 1. Map the user's spec.scheduling (here: gang with no minCount, pinned to a
	//    single node per topology) into the library IR. A nil sub-field is left
	//    nil so it can fall back to the controller default field-by-field.
	userConfig := workloadbuilder.MapPodGroupConfig(workloadbuilder.WorkloadPodGroupConfig{
		Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}, // minCount defaulted below
		},
		Constraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
			Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "kubernetes.io/hostname"}},
		},
		// DisruptionMode and ResourceClaims left unset: use the defaults.
	})

	// 2. Assemble the single-node workload tree: a controller default (Basic),
	//    the user's intent layered on top, and a callback that defaults the gang
	//    minCount to the Job's parallelism when the user left it unset. The
	//    callback mutates only the resolved config, never the caller's inputs.
	item := &workloadbuilder.WorkloadItem{
		Name:          "trainer-pgt-0",
		DefaultConfig: &workloadbuilder.SchedulingConfig{Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}}},
		UserConfig:    userConfig,
		Callbacks: []workloadbuilder.SchedulingConfigFunc{
			func(cfg *workloadbuilder.SchedulingConfig) {
				if g := cfg.Policy.Gang; g != nil && g.MinCount == nil {
					g.MinCount = new(parallelism)
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
	})

	// 4. Run the complex controller-policy checks. In-tree controllers (i.e. Job)
	//    get structural building-block validation from declarative validation at
	//    the apiserver, out-of-tree controllers must also call the generated
	//    schedulingv1alpha3.Validate_* functions first.
	if errs := builder.Validate(field.NewPath("spec", "scheduling")); len(errs) > 0 {
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
