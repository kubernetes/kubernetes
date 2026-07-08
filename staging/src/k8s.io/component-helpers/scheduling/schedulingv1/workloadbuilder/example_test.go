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

package workloadbuilder_test

import (
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
)

// Example_jobControllerE2E shows the end-to-end flow a controller uses to turn its
// user-facing scheduling API into a scheduler-facing Workload and a runtime
// PodGroup. It mirrors what the built-in Job controller does for a single Job:
// map spec.scheduling, layer it over a controller default, default the gang
// minCount to the Job's parallelism, compile the Workload, then materialize the
// PodGroup from the compiled template.
func Example_jobControllerE2E() {
	// The Job's parallelism, used to default an unset gang minCount.
	parallelism := int32(4)

	// 1. Map the user's spec.scheduling (here: gang with no minCount, pinned to a
	//    single node per topology) into the library IR. A nil sub-field is left
	//    nil so it can fall back to the controller default field-by-field.
	userConfig := workloadbuilder.MapPodGroupConfig(
		&schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}, // minCount defaulted below
		},
		&schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
			Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "kubernetes.io/hostname"}},
		},
		nil, // disruption mode: use the default
		nil, // resource claims: none
	)

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

	// 3. Compile the Workload. The owner becomes spec.controllerRef, which the
	//    controller later uses to discover the Workload and which drives garbage
	//    collection.
	owner := &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "trainer", UID: "job-uid"}
	workload, err := workloadbuilder.Build(item, workloadbuilder.BuildOptions{
		Name:      "trainer-wl",
		Namespace: "ml",
		Owner:     owner,
	})
	if err != nil {
		fmt.Printf("build failed: %v\n", err)
		return
	}

	// 4. Materialize the runtime PodGroup from the compiled template. Owners are
	//    caller-supplied; a root Job owns its PodGroup.
	podGroup, err := workloadbuilder.NewPodGroup(workload, item.Name, "trainer-pg", []metav1.OwnerReference{*owner})
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
