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

package pods

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	dynamicvolumesfeature "k8s.io/component-helpers/nodedeclaredfeatures/features/dynamicvolumes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDynamicNodeLocalEphemeralVolumesSubresource(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.NodeDeclaredFeatures:             true,
		features.DynamicNodeLocalEphemeralVolumes: true,
	})
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-dynamic-volumes", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx := context.Background()
	nodeName := "dynamic-vol-node"
	restartPolicy := &v1.ContainerDynamicVolumePolicy{
		RestartPolicy: v1.DynamicVolumeRestartContainer,
	}

	createNode := func(t *testing.T, declaredFeatures []string) {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
			},
			Status: v1.NodeStatus{
				NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.36.0"},
				DeclaredFeatures: declaredFeatures,
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("8Gi"),
				},
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("8Gi"),
				},
			},
		}
		if _, err := client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create node: %v", err)
		}
		t.Cleanup(func() {
			_ = client.CoreV1().Nodes().Delete(ctx, nodeName, metav1.DeleteOptions{})
		})
	}

	basePod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "dyn-vol-pod-",
				Namespace:    ns.Name,
			},
			Spec: v1.PodSpec{
				NodeName: nodeName,
				Volumes: []v1.Volume{
					{
						Name: "vol-init",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:                "c1",
						Image:               "busybox",
						DynamicVolumePolicy: restartPolicy,
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol-init",
								MountPath: "/mnt/init",
							},
						},
					},
				},
			},
		}
	}

	markPodRunningAndInitialized := func(t *testing.T, pod *v1.Pod) *v1.Pod {
		t.Helper()
		withStatus := pod.DeepCopy()
		withStatus.Status.Phase = v1.PodRunning
		withStatus.Status.Conditions = []v1.PodCondition{
			{
				Type:   v1.PodInitialized,
				Status: v1.ConditionTrue,
			},
		}
		updated, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, withStatus, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to mark pod running and initialized: %v", err)
		}
		return updated
	}

	updateDynamic := func(ctx context.Context, pod *v1.Pod) (*v1.Pod, error) {
		body, err := json.Marshal(pod)
		if err != nil {
			return nil, err
		}
		result := &v1.Pod{}
		err = client.CoreV1().RESTClient().Put().
			Namespace(ns.Name).
			Resource("pods").
			Name(pod.Name).
			SubResource("dynamic").
			Body(body).
			Do(ctx).
			Into(result)
		return result, err
	}

	t.Run("rejects dynamic volume mutation when node does not declare feature", func(t *testing.T) {
		createNode(t, []string{"SomeOtherFeature"})
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, basePod(), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		defer func() {
			_ = client.CoreV1().Pods(ns.Name).Delete(ctx, created.Name, metav1.DeleteOptions{})
		}()
		created = markPodRunningAndInitialized(t, created)

		err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			updated := created.DeepCopy()
			updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
				Name: "vol-cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: "my-cm"},
					},
				},
			})
			updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
				Name:      "vol-cm",
				MountPath: "/mnt/cm",
			})
			_, updateErr := updateDynamic(ctx, updated)
			if updateErr != nil && strings.Contains(updateErr.Error(), "DynamicNodeLocalEphemeralVolumes") {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("expected NodeDeclaredFeature rejection for DynamicNodeLocalEphemeralVolumes, got: %v", err)
		}
	})

	t.Run("allows adding and removing node-local ephemeral volumes via /dynamic when node declares feature", func(t *testing.T) {
		createNode(t, []string{dynamicvolumesfeature.Feature.Name()})
		podSpec := basePod()
		podSpec.Spec.Volumes = append(podSpec.Spec.Volumes, v1.Volume{
			Name: "vol-hostpath",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp/static-hostpath"},
			},
		})
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, podSpec, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		defer func() {
			_ = client.CoreV1().Pods(ns.Name).Delete(ctx, created.Name, metav1.DeleteOptions{})
		}()
		// Supported volumes (vol-init) are seeded as Pending on create; unsupported volumes (vol-hostpath) are excluded.
		if len(created.Status.VolumeStatuses) != 1 ||
			created.Status.VolumeStatuses[0].Name != "vol-init" ||
			created.Status.VolumeStatuses[0].HostPhase != v1.PodVolumeHostPending {
			t.Fatalf("expected [vol-init:Pending] on pod create, got: %+v", created.Status.VolumeStatuses)
		}
		created = markPodRunningAndInitialized(t, created)

		var afterAdd *v1.Pod
		err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			updated := created.DeepCopy()
			updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
				Name: "vol-cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: "my-cm"},
					},
				},
			})
			updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
				Name:      "vol-cm",
				MountPath: "/mnt/cm",
			})
			var updateErr error
			afterAdd, updateErr = updateDynamic(ctx, updated)
			return updateErr == nil, nil
		})
		if err != nil {
			t.Fatalf("unexpected error adding configMap volume via /dynamic: %v", err)
		}
		if len(afterAdd.Spec.Volumes) != 3 {
			t.Fatalf("expected 3 volumes after /dynamic update, got %d", len(afterAdd.Spec.Volumes))
		}
		if len(afterAdd.Status.VolumeStatuses) != 2 ||
			afterAdd.Status.VolumeStatuses[0].Name != "vol-cm" ||
			afterAdd.Status.VolumeStatuses[0].HostPhase != v1.PodVolumeHostPending {
			t.Fatalf("expected /dynamic add to seed vol-cm as Pending in VolumeStatuses, got: %+v", afterAdd.Status.VolumeStatuses)
		}

		// Rejects adding a non-node-local volume (e.g. PVC) via /dynamic.
		withPVC := afterAdd.DeepCopy()
		withPVC.Spec.Volumes = append(withPVC.Spec.Volumes, v1.Volume{
			Name: "vol-pvc",
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "my-pvc"},
			},
		})
		if _, err := updateDynamic(ctx, withPVC); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when adding PVC volume via /dynamic, got: %v", err)
		}

		// Rejects mutating volumeMounts referencing an unsupported static volume (hostPath).
		withHostPathMount := afterAdd.DeepCopy()
		withHostPathMount.Spec.Containers[0].VolumeMounts = append(withHostPathMount.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      "vol-hostpath",
			MountPath: "/mnt/hostpath",
		})
		if _, err := updateDynamic(ctx, withHostPathMount); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when adding volumeMount for unsupported hostPath volume via /dynamic, got: %v", err)
		}

		// Verify PodStatus.VolumeStatuses can be updated via /status.
		withStatus := afterAdd.DeepCopy()
		withStatus.Status.VolumeStatuses = []v1.PodVolumeStatus{
			{Name: "vol-init", HostPhase: v1.PodVolumeHostMounted},
			{Name: "vol-cm", HostPhase: v1.PodVolumeHostPending},
		}
		statusUpdated, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, withStatus, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("unexpected error updating PodStatus.VolumeStatuses: %v", err)
		}
		if len(statusUpdated.Status.VolumeStatuses) != 2 {
			t.Fatalf("expected 2 VolumeStatuses in PodStatus, got %d", len(statusUpdated.Status.VolumeStatuses))
		}

		// Remove vol-cm via /dynamic before kubelet mounts it (vol-cm is still Pending in VolumeStatuses).
		afterRemove := statusUpdated.DeepCopy()
		afterRemove.Spec.Volumes = afterRemove.Spec.Volumes[:2]
		afterRemove.Spec.Containers[0].VolumeMounts = afterRemove.Spec.Containers[0].VolumeMounts[:1]
		removedPod, err := updateDynamic(ctx, afterRemove)
		if err != nil {
			t.Fatalf("unexpected error removing volume via /dynamic: %v", err)
		}
		if len(removedPod.Spec.Volumes) != 2 {
			t.Fatalf("expected 2 volumes after removal via /dynamic, got %d", len(removedPod.Spec.Volumes))
		}

		// Rapidly re-adding vol-cm with different parameters before kubelet clears VolumeStatuses is rejected.
		rapidReAdd := removedPod.DeepCopy()
		rapidReAdd.Spec.Volumes = append(rapidReAdd.Spec.Volumes, v1.Volume{
			Name: "vol-cm",
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{Name: "other-cm"},
				},
			},
		})
		if _, err := updateDynamic(ctx, rapidReAdd); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when rapidly re-adding vol-cm while still Pending in VolumeStatuses, got: %v", err)
		}

		// A stale status update (ObservedGeneration < removedPod.Generation) must preserve vol-cm=Pending.
		staleStatus := removedPod.DeepCopy()
		staleStatus.Status.ObservedGeneration = 1
		staleStatus.Status.VolumeStatuses = []v1.PodVolumeStatus{
			{Name: "vol-init", HostPhase: v1.PodVolumeHostMounted},
		}
		afterStaleStatus, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, staleStatus, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("unexpected error updating stale status: %v", err)
		}
		if len(afterStaleStatus.Status.VolumeStatuses) != 2 {
			t.Fatalf("expected stale status update to preserve Pending vol-cm, got: %+v", afterStaleStatus.Status.VolumeStatuses)
		}

		// Once kubelet updates status at removedPod.Generation without vol-cm, re-adding vol-cm with other-cm succeeds.
		clearedStatus := afterStaleStatus.DeepCopy()
		clearedStatus.Status.ObservedGeneration = removedPod.Generation
		clearedStatus.Status.VolumeStatuses = []v1.PodVolumeStatus{
			{Name: "vol-init", HostPhase: v1.PodVolumeHostMounted},
		}
		afterClearedStatus, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, clearedStatus, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("unexpected error clearing status at current generation: %v", err)
		}
		reAddAfterClear := afterClearedStatus.DeepCopy()
		reAddAfterClear.Spec.Volumes = append(reAddAfterClear.Spec.Volumes, v1.Volume{
			Name: "vol-cm",
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{Name: "other-cm"},
				},
			},
		})
		if _, err := updateDynamic(ctx, reAddAfterClear); err != nil {
			t.Fatalf("expected re-adding vol-cm to succeed once cleared at current generation, got: %v", err)
		}
	})
}
