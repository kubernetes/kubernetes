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

package controller

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

var (
	productionLabel         = map[string]string{"type": "production"}
	testLabel               = map[string]string{"type": "testing"}
	productionLabelSelector = labels.Set{"type": "production"}.AsSelector()
	controllerUID           = "123"
)

func newPod(podName string, label map[string]string, owner metav1.Object) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Labels:    label,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image: "foo/bar",
				},
			},
		},
	}
	if owner != nil {
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, apps.SchemeGroupVersion.WithKind("Fake"))}
	}
	return pod
}

func TestClaimPods(t *testing.T) {
	controllerKind := schema.GroupVersionKind{}
	type test struct {
		name    string
		manager *PodControllerRefManager
		pods    []*v1.Pod
		claimed []*v1.Pod
		patches int
	}
	var tests = []test{
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			return test{
				name: "Claim pods with correct label",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, nil), newPod("pod2", testLabel, nil)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, nil)},
				patches: 1,
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			now := metav1.Now()
			controller.DeletionTimestamp = &now
			return test{
				name: "Controller marked for deletion can not claim pods",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, nil), newPod("pod2", productionLabel, nil)},
				claimed: nil,
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			now := metav1.Now()
			controller.DeletionTimestamp = &now
			return test{
				name: "Controller marked for deletion can not claim new pods",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", productionLabel, nil)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller2 := v1.ReplicationController{}
			controller.UID = types.UID(controllerUID)
			controller.Namespace = metav1.NamespaceDefault
			controller2.UID = types.UID("AAAAA")
			controller2.Namespace = metav1.NamespaceDefault
			return test{
				name: "Controller can not claim pods owned by another controller",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", productionLabel, &controller2)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			return test{
				name: "Controller releases claimed pods when selector doesn't match",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", testLabel, &controller)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller)},
				patches: 1,
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			podToDelete1 := newPod("pod1", productionLabel, &controller)
			podToDelete2 := newPod("pod2", productionLabel, nil)
			now := metav1.Now()
			podToDelete1.DeletionTimestamp = &now
			podToDelete2.DeletionTimestamp = &now

			return test{
				name: "Controller does not claim orphaned pods marked for deletion",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{podToDelete1, podToDelete2},
				claimed: []*v1.Pod{podToDelete1},
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			return test{
				name: "Controller claims or release pods according to selector with finalizers",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil },
					"foo-finalizer", "bar-finalizer"),
				pods:    []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod2", testLabel, &controller), newPod("pod3", productionLabel, nil)},
				claimed: []*v1.Pod{newPod("pod1", productionLabel, &controller), newPod("pod3", productionLabel, nil)},
				patches: 2,
			}
		}(),
		func() test {
			controller := v1.ReplicationController{}
			controller.Namespace = metav1.NamespaceDefault
			controller.UID = types.UID(controllerUID)
			pod1 := newPod("pod1", productionLabel, nil)
			pod2 := newPod("pod2", productionLabel, nil)
			pod2.Namespace = "fakens"
			return test{
				name: "Controller does not claim pods of different namespace",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{pod1, pod2},
				claimed: []*v1.Pod{pod1},
				patches: 1,
			}
		}(),
		func() test {
			// act as a cluster-scoped controller
			controller := v1.ReplicationController{}
			controller.Namespace = ""
			controller.UID = types.UID(controllerUID)
			pod1 := newPod("pod1", productionLabel, nil)
			pod2 := newPod("pod2", productionLabel, nil)
			pod2.Namespace = "fakens"
			return test{
				name: "Cluster scoped controller claims pods of specified namespace",
				manager: NewPodControllerRefManager(&FakePodControl{},
					&controller,
					productionLabelSelector,
					controllerKind,
					func(ctx context.Context) error { return nil }),
				pods:    []*v1.Pod{pod1, pod2},
				claimed: []*v1.Pod{pod1, pod2},
				patches: 2,
			}
		}(),
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			claimed, err := test.manager.ClaimPods(context.TODO(), test.pods)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(test.claimed, claimed); diff != "" {
				t.Errorf("Claimed wrong pods (-want,+got):\n%s", diff)
			}
			fakePodControl, ok := test.manager.podControl.(*FakePodControl)
			if !ok {
				return
			}
			if p := len(fakePodControl.Patches); p != test.patches {
				t.Errorf("ClaimPods issues %d patches, want %d", p, test.patches)
			}
			for _, p := range fakePodControl.Patches {
				patch := string(p)
				if uid := string(test.manager.Controller.GetUID()); !strings.Contains(patch, uid) {
					t.Errorf("Patch doesn't contain controller UID %s", uid)
				}
				for _, f := range test.manager.finalizers {
					if !strings.Contains(patch, f) {
						t.Errorf("Patch doesn't contain finalizer %s, %q", patch, f)
					}
				}
			}
		})
	}
}

func TestGeneratePatchBytesForDelete(t *testing.T) {
	tests := []struct {
		name         string
		ownerUID     []types.UID
		dependentUID types.UID
		finalizers   []string
		want         []byte
	}{
		{
			name:         "check the structure of patch bytes",
			ownerUID:     []types.UID{"ss1"},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"}]}}`),
		},
		{
			name:         "check if parent uid is escaped",
			ownerUID:     []types.UID{`ss1"hello`},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1\"hello"}]}}`),
		},
		{
			name:         "check if revision uid uid is escaped",
			ownerUID:     []types.UID{`ss1`},
			dependentUID: `ss2"hello`,
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2\"hello","ownerReferences":[{"$patch":"delete","uid":"ss1"}]}}`),
		},
		{
			name:         "check the structure of patch bytes with multiple owners",
			ownerUID:     []types.UID{"ss1", "ss2"},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"},{"$patch":"delete","uid":"ss2"}]}}`),
		},
		{
			name:         "check the structure of patch bytes with a finalizer and multiple owners",
			ownerUID:     []types.UID{"ss1", "ss2"},
			dependentUID: "ss2",
			finalizers:   []string{"f1"},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"},{"$patch":"delete","uid":"ss2"}],"$deleteFromPrimitiveList/finalizers":["f1"]}}`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := GenerateDeleteOwnerRefStrategicMergeBytes(tt.dependentUID, tt.ownerUID, tt.finalizers...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("generatePatchBytesForDelete() got = %s, want %s", got, tt.want)
			}
		})
	}
}
