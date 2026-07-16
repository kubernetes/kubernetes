/*
Copyright 2025 The Kubernetes Authors.

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

package podgroup

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// createBoundRWOPPVC creates a HostPath PV and a PVC bound to it, both with the
// ReadWriteOncePod access mode.
func createBoundRWOPPVC(cs kubernetes.Interface, ns, name string) error {
	storage := v1.VolumeResourceRequirements{Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Mi")}}
	volType := v1.HostPathDirectoryOrCreate
	pv, err := testutils.CreatePV(cs, st.MakePersistentVolume().
		Name(fmt.Sprintf("pv-%s-%s", ns, name)).
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Capacity(storage.Requests).
		HostPathVolumeSource(&v1.HostPathVolumeSource{Path: "/mnt", Type: &volType}).
		Obj())
	if err != nil {
		return fmt.Errorf("cannot create pv: %w", err)
	}
	_, err = testutils.CreatePVC(cs, st.MakePersistentVolumeClaim().
		Name(name).
		Namespace(ns).
		// Annotation and volume name required for PVC to be considered bound.
		Annotation(volume.AnnBindCompleted, "true").
		VolumeName(pv.Name).
		AccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod}).
		Resources(storage).
		Obj())
	if err != nil {
		return fmt.Errorf("cannot create pvc: %w", err)
	}
	return nil
}

// TestPodGroupSchedulingWithReadWriteOncePodPVC is a regression test for the
// snapshot AssumePod/ForgetPod bug. During a PodGroup scheduling cycle the pods
// of a group are assumed into the scheduler snapshot one by one, so AssumePod
// must record the assumed pod's PVCs in the snapshot's usedPVCRefCounts:
// otherwise the VolumeRestrictions plugin does not see that a ReadWriteOncePod
// PVC is already used by a pod assumed earlier in the same cycle and silently
// lets a second pod mount it.
//
// The first pod is created and enqueued first so the pod group cycle (which
// orders pods of equal priority by enqueue time) always assumes it before the
// second pod, although the conflict is symmetric: whichever pod is assumed
// second must observe the PVC of the one assumed first. In the conflict case
// both pods reference the same ReadWriteOncePod PVC, so the group must stay
// unschedulable; without the fix both pods are scheduled and share the PVC in
// violation of its access mode. The no-conflict case is the positive control:
// with two distinct PVCs the same group schedules, proving the setup is
// schedulable and the conflict case fails for the intended reason.
func TestPodGroupSchedulingWithReadWriteOncePodPVC(t *testing.T) {
	nodeCapacity := map[v1.ResourceName]string{v1.ResourceCPU: "2"}
	podRequest := map[v1.ResourceName]string{v1.ResourceCPU: "1"}

	workload := st.MakeWorkload().Name("workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t").MinCount(2).Obj()).Obj()
	podGroup := st.MakePodGroup().Name("pg").WorkloadRef("t", "workload").Priority(200).MinCount(2).Obj()

	tests := []struct {
		name string
		// secondPodPVC is the PVC referenced by the pod assumed second in the
		// cycle; the pod assumed first always references pvc-1.
		secondPodPVC string
		steps        []stepsframework.Step
	}{
		{
			name:         "two pods sharing a ReadWriteOncePod PVC keep the group unschedulable",
			secondPodPVC: "pvc-1",
			steps: []stepsframework.Step{
				{
					Name:                     "Verify the group becomes unschedulable instead of sharing the PVC",
					WaitForPodsUnschedulable: []string{"pvc-pod-1", "pvc-pod-2"},
				},
				{
					Name: "Verify PodGroup condition is set to Unschedulable",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonUnschedulable,
					},
				},
			},
		},
		{
			name:         "two pods with distinct ReadWriteOncePod PVCs are scheduled",
			secondPodPVC: "pvc-2",
			steps: []stepsframework.Step{
				{
					Name:                 "Verify both pods are scheduled",
					WaitForPodsScheduled: []string{"pvc-pod-1", "pvc-pod-2"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-pvc",
				// disable backoff
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			ns := testCtx.NS.Name

			for _, pvcName := range sets.List(sets.New("pvc-1", tt.secondPodPVC)) {
				if err := createBoundRWOPPVC(testCtx.ClientSet, ns, pvcName); err != nil {
					t.Fatal(err)
				}
			}

			// firstPod is assumed first (created and enqueued before the second
			// pod) and takes pvc-1.
			firstPod := st.MakePod().Name("pvc-pod-1").
				Req(podRequest).Container("image").
				PVC("pvc-1").
				PodGroupName("pg").Priority(200).Obj()
			// secondPod is assumed second; whether its PVC conflicts with the
			// first pod's depends on the test case.
			secondPod := st.MakePod().Name("pvc-pod-2").
				Req(podRequest).Container("image").
				PVC(tt.secondPodPVC).
				PodGroupName("pg").Priority(200).Obj()

			commonSteps := []stepsframework.Step{
				{
					Name:        "Create Nodes",
					CreateNodes: []*v1.Node{st.MakeNode().Name("node-1").Capacity(nodeCapacity).Obj()},
				},
				{
					Name:            "Create workloads",
					CreateWorkloads: []*schedulingapi.Workload{workload},
				},
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: podGroup,
				},
				{
					Name:              "Create both pods belonging to the group",
					CreatePodsInOrder: []*v1.Pod{firstPod, secondPod},
				},
			}

			if err := stepsframework.RunSteps(testCtx, t, ns, append(commonSteps, tt.steps...)); err != nil {
				t.Fatal(err)
			}
		})
	}
}
