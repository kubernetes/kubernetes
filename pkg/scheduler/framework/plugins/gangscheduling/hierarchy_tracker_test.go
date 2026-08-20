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

package gangscheduling

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/utils/ptr"
)

func TestHierarchyTracker_GangQuorum(t *testing.T) {
	tracker := NewHierarchyTracker()

	cpg := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "root-cpg",
		},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
		},
	}
	tracker.OnCompositePodGroupAdd(cpg)

	pg := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "child-pg",
		},
		Spec: schedulingv1beta1.PodGroupSpec{
			ParentCompositePodGroupName: ptr.To("root-cpg"),
			SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{
					MinCount: 2,
				},
			},
		},
	}
	tracker.OnPodGroupAdd(pg)

	cpgKey := fwk.CompositePodGroupKey("default", "root-cpg")

	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 0 {
		t.Fatalf("expected 0 ready children initially, got %d", rc)
	}

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-1",
		},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{
				PodGroupName: ptr.To("child-pg"),
			},
		},
	}
	tracker.OnPodAdd(pod1)

	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 0 {
		t.Fatalf("expected 0 ready children after 1st pod (minCount=2), got %d", rc)
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-2",
		},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{
				PodGroupName: ptr.To("child-pg"),
			},
		},
	}
	tracker.OnPodAdd(pod2)

	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 1 {
		t.Fatalf("expected 1 ready child after 2nd pod, got %d", rc)
	}

	tracker.OnPodDelete(pod2)
	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 0 {
		t.Fatalf("expected 0 ready children after deleting pod, got %d", rc)
	}
}

func TestHierarchyTracker_BasicPolicy(t *testing.T) {
	tracker := NewHierarchyTracker()

	cpg := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "root-basic-cpg",
		},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
			},
		},
	}
	tracker.OnCompositePodGroupAdd(cpg)

	pg := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "child-basic-pg",
		},
		Spec: schedulingv1beta1.PodGroupSpec{
			ParentCompositePodGroupName: ptr.To("root-basic-cpg"),
			SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
				Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
			},
		},
	}
	tracker.OnPodGroupAdd(pg)

	cpgKey := fwk.CompositePodGroupKey("default", "root-basic-cpg")

	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 0 {
		t.Fatalf("expected 0 ready children initially for basic policy, got %d", rc)
	}

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-basic-1",
		},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{
				PodGroupName: ptr.To("child-basic-pg"),
			},
		},
	}
	tracker.OnPodAdd(pod1)

	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 1 {
		t.Fatalf("expected 1 ready child after 1st pod for basic policy, got %d", rc)
	}

	tracker.OnPodDelete(pod1)
	if rc := tracker.ReadyChildrenCount(cpgKey); rc != 0 {
		t.Fatalf("expected 0 ready children after deleting pod for basic policy, got %d", rc)
	}
}
