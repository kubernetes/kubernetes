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
	"slices"
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
	dynamiccontainersfeature "k8s.io/component-helpers/nodedeclaredfeatures/features/dynamiccontainers"
	dynamicvolumesfeature "k8s.io/component-helpers/nodedeclaredfeatures/features/dynamicvolumes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDynamicNodeLocalEphemeralVolumesSubresource(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-dyn-vols", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx := context.Background()
	nodeName := "dynamic-vol-node"

	createNode := func(t *testing.T, declaredFeatures []string) {
		sortedFeatures := slices.Clone(declaredFeatures)
		slices.Sort(sortedFeatures)
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
			},
			Status: v1.NodeStatus{
				NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
				DeclaredFeatures: sortedFeatures,
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
						Name:  "c1",
						Image: "busybox",
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
		createNode(t, []string{dynamiccontainersfeature.Feature.Name()})
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
			updated.Spec.Containers = append(updated.Spec.Containers, v1.Container{
				Name:  "c2",
				Image: "busybox",
				VolumeMounts: []v1.VolumeMount{{
					Name:      "vol-cm",
					MountPath: "/mnt/cm",
				}},
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

	t.Run("rejects adding containers when DynamicContainers feature gate is disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, false)
		createNode(t, []string{dynamicvolumesfeature.Feature.Name(), dynamiccontainersfeature.Feature.Name()})
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, basePod(), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		defer func() {
			_ = client.CoreV1().Pods(ns.Name).Delete(ctx, created.Name, metav1.DeleteOptions{})
		}()
		created = markPodRunningAndInitialized(t, created)

		var afterVolAdd *v1.Pod
		err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			updated := created.DeepCopy()
			updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
				Name:         "vol-ephem-only",
				VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
			})
			var updateErr error
			afterVolAdd, updateErr = updateDynamic(ctx, updated)
			return updateErr == nil, nil
		})
		if err != nil {
			t.Fatalf("expected adding volume with DynamicContainers=false to succeed, got: %v", err)
		}

		withAddedContainer := afterVolAdd.DeepCopy()
		withAddedContainer.Spec.Containers = append(withAddedContainer.Spec.Containers, v1.Container{
			Name:  "c2",
			Image: "busybox",
		})
		if _, err := updateDynamic(ctx, withAddedContainer); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when adding container via /dynamic with DynamicContainers disabled, got: %v", err)
		}
	})

	t.Run("allows adding and removing node-local ephemeral volumes with dynamic containers via /dynamic", func(t *testing.T) {
		createNode(t, []string{dynamicvolumesfeature.Feature.Name(), dynamiccontainersfeature.Feature.Name()})
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
		if len(created.Status.VolumeStatuses) != 0 {
			t.Fatalf("expected empty VolumeStatuses on pod create, got: %+v", created.Status.VolumeStatuses)
		}
		// Reordering spec.volumes on /dynamic while the pod is still Pending must be rejected by lifecycle validation.
		reorderedPending := created.DeepCopy()
		reorderedPending.Spec.Volumes[0], reorderedPending.Spec.Volumes[1] = reorderedPending.Spec.Volumes[1], reorderedPending.Spec.Volumes[0]
		if _, err := updateDynamic(ctx, reorderedPending); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when reordering volumes while pod is Pending, got: %v", err)
		}
		created = markPodRunningAndInitialized(t, created)

		// Adding an unreferenced volume via /dynamic does not seed Pending in VolumeStatuses.
		var afterUnreferenced *v1.Pod
		err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			updated := created.DeepCopy()
			updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
				Name: "vol-unreferenced",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			})
			var updateErr error
			afterUnreferenced, updateErr = updateDynamic(ctx, updated)
			return updateErr == nil, nil
		})
		if err != nil {
			t.Fatalf("unexpected error adding unreferenced emptyDir volume via /dynamic: %v", err)
		}
		if len(afterUnreferenced.Status.VolumeStatuses) != 0 {
			t.Fatalf("expected unreferenced dynamic volume not to seed Pending in VolumeStatuses, got: %+v", afterUnreferenced.Status.VolumeStatuses)
		}

		// Rejects mutating VolumeMounts on an existing container via /dynamic.
		mutatedExistingMounts := afterUnreferenced.DeepCopy()
		mutatedExistingMounts.Spec.Containers[0].VolumeMounts = append(mutatedExistingMounts.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      "vol-unreferenced",
			MountPath: "/mnt/unreferenced",
		})
		if _, err := updateDynamic(ctx, mutatedExistingMounts); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when mutating volumeMounts on existing container, got: %v", err)
		}

		// Add a configMap volume together with dynamic container c2 mounting it.
		updated := afterUnreferenced.DeepCopy()
		updated.Spec.Volumes = append(updated.Spec.Volumes, v1.Volume{
			Name: "vol-cm",
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{Name: "my-cm"},
				},
			},
		})
		updated.Spec.Containers = append(updated.Spec.Containers, v1.Container{
			Name:  "c2",
			Image: "busybox",
			VolumeMounts: []v1.VolumeMount{{
				Name:      "vol-cm",
				MountPath: "/mnt/cm",
			}},
		})
		afterAdd, err := updateDynamic(ctx, updated)
		if err != nil {
			t.Fatalf("unexpected error adding configMap volume and dynamic container c2 via /dynamic: %v", err)
		}
		if len(afterAdd.Spec.Volumes) != 4 {
			t.Fatalf("expected 4 volumes after /dynamic update, got %d", len(afterAdd.Spec.Volumes))
		}
		if len(afterAdd.Status.VolumeStatuses) != 0 {
			t.Fatalf("expected /dynamic add not to mutate VolumeStatuses before kubelet status sync, got: %+v", afterAdd.Status.VolumeStatuses)
		}

		// Removing a newly added mounted volume + container before kubelet reports statuses is rejected.
		prematureRemoveAdded := afterAdd.DeepCopy()
		prematureRemoveAdded.Spec.Volumes = prematureRemoveAdded.Spec.Volumes[:3]
		prematureRemoveAdded.Spec.Containers = prematureRemoveAdded.Spec.Containers[:1]
		if _, err := updateDynamic(ctx, prematureRemoveAdded); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when removing c2 and vol-cm before kubelet reports statuses, got: %v", err)
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

		// Verify PodStatus.VolumeStatuses supports both HostPhase and CSIVolumeHealth HealthStatus co-existing via /status.
		withStatus := afterAdd.DeepCopy()
		withStatus.Status.ContainerStatuses = []v1.ContainerStatus{
			{Name: "c1", State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
			{Name: "c2", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: "ContainerCreating"}}},
		}
		withStatus.Status.VolumeStatuses = []v1.PodVolumeStatus{
			{
				Name: "vol-hostpath",
				HealthStatus: &v1.VolumeHealthStatus{
					HealthConditions: []v1.VolumeHealthCondition{{
						Status: v1.VolumeHealthHealthy,
						Reason: "VolumeHealthy",
					}},
				},
			},
			{Name: "vol-init", HostPhase: v1.PodVolumeHostMounted},
			{Name: "vol-cm", HostPhase: v1.PodVolumeHostPending},
		}
		statusUpdated, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, withStatus, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("unexpected error updating PodStatus.VolumeStatuses: %v", err)
		}

		// Removing vol-cm while c2 still references it in spec.containers is rejected.
		removeVolKeepCtr := statusUpdated.DeepCopy()
		removeVolKeepCtr.Spec.Volumes = removeVolKeepCtr.Spec.Volumes[:3]
		if _, err := updateDynamic(ctx, removeVolKeepCtr); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when removing vol-cm while c2 still references it, got: %v", err)
		}

		// Jointly remove c2 and vol-cm via /dynamic once status has been reported.
		afterRemove := statusUpdated.DeepCopy()
		afterRemove.Spec.Volumes = afterRemove.Spec.Volumes[:3]
		afterRemove.Spec.Containers = afterRemove.Spec.Containers[:1]
		removedPod, err := updateDynamic(ctx, afterRemove)
		if err != nil {
			t.Fatalf("unexpected error jointly removing c2 and vol-cm via /dynamic: %v", err)
		}
		if len(removedPod.Spec.Volumes) != 3 {
			t.Fatalf("expected 3 volumes after removal via /dynamic, got %d", len(removedPod.Spec.Volumes))
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
		rapidReAdd.Spec.Containers = append(rapidReAdd.Spec.Containers, v1.Container{
			Name:  "c3",
			Image: "busybox",
			VolumeMounts: []v1.VolumeMount{{
				Name:      "vol-cm",
				MountPath: "/mnt/cm",
			}},
		})
		if _, err := updateDynamic(ctx, rapidReAdd); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when rapidly re-adding vol-cm while still Pending in VolumeStatuses, got: %v", err)
		}

		// Once kubelet updates status without vol-cm, re-adding vol-cm with other-cm succeeds
		// even if a Terminated container status still lists vol-cm in its historical VolumeMounts.
		clearedStatus := removedPod.DeepCopy()
		clearedStatus.Status.ObservedGeneration = removedPod.Generation
		clearedStatus.Status.VolumeStatuses = []v1.PodVolumeStatus{
			{Name: "vol-init", HostPhase: v1.PodVolumeHostMounted},
		}
		clearedStatus.Status.ContainerStatuses = []v1.ContainerStatus{
			{
				Name:  "c1",
				State: v1.ContainerState{Running: &v1.ContainerStateRunning{}},
				VolumeMounts: []v1.VolumeMountStatus{
					{Name: "vol-init", MountPath: "/mnt/init"},
				},
			},
			{
				Name:  "c2",
				State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}},
				VolumeMounts: []v1.VolumeMountStatus{
					{Name: "vol-cm", MountPath: "/mnt/cm"},
				},
			},
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
		reAddAfterClear.Spec.Containers = append(reAddAfterClear.Spec.Containers, v1.Container{
			Name:  "c3",
			Image: "busybox",
			VolumeMounts: []v1.VolumeMount{{
				Name:      "vol-cm",
				MountPath: "/mnt/cm",
			}},
		})
		afterReAdd, err := updateDynamic(ctx, reAddAfterClear)
		if err != nil {
			t.Fatalf("expected re-adding vol-cm with c3 to succeed once cleared at current generation, got: %v", err)
		}

		// Reordering volumes is rejected on pods and pods/resize, and allowed on pods/dynamic.
		reordered := afterReAdd.DeepCopy()
		last := len(reordered.Spec.Volumes) - 1
		reordered.Spec.Volumes[0], reordered.Spec.Volumes[last] = reordered.Spec.Volumes[last], reordered.Spec.Volumes[0]
		if _, err := client.CoreV1().Pods(ns.Name).Update(ctx, reordered, metav1.UpdateOptions{}); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error reordering volumes on pods endpoint, got: %v", err)
		}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateResize(ctx, reordered.Name, reordered, metav1.UpdateOptions{}); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error reordering volumes on pods/resize endpoint, got: %v", err)
		}
		afterReorder, err := updateDynamic(ctx, reordered)
		if err != nil {
			t.Fatalf("expected reordering volumes on pods/dynamic to succeed, got: %v", err)
		}
		if afterReorder.Spec.Volumes[0].Name != "vol-cm" || afterReorder.Spec.Volumes[last].Name != "vol-init" {
			t.Fatalf("unexpected volume order after pods/dynamic reorder: %+v", afterReorder.Spec.Volumes)
		}
	})
}

func TestDynamicNodeLocalEphemeralVolumesAdmission(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeBindMountOptions, true)
	flags := []string{
		"--endpoint-reconciler-type=none",
		"--enable-admission-plugins=ServiceAccount,AlwaysPullImages,PodSecurity",
	}
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-dyn-vol-adm", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx := context.Background()
	nodeName := "dyn-adm-node"
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		Status: v1.NodeStatus{
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
			DeclaredFeatures: []string{dynamiccontainersfeature.Feature.Name(), dynamicvolumesfeature.Feature.Name()},
		},
	}
	if _, err := client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	sa := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "restricted-sa",
			Namespace: ns.Name,
			Annotations: map[string]string{
				"kubernetes.io/enforce-mountable-secrets": "true",
			},
		},
		Secrets: []v1.ObjectReference{{Name: "allowed-secret"}},
	}
	if _, err := client.CoreV1().ServiceAccounts(ns.Name).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create ServiceAccount: %v", err)
	}

	created, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "adm-pod",
			Namespace: ns.Name,
		},
		Spec: v1.PodSpec{
			NodeName:           nodeName,
			ServiceAccountName: sa.Name,
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "busybox:1.36",
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}

	withStatus := created.DeepCopy()
	withStatus.Status.Phase = v1.PodRunning
	withStatus.Status.Conditions = []v1.PodCondition{{Type: v1.PodInitialized, Status: v1.ConditionTrue}}
	created, err = client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, withStatus, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to mark pod running: %v", err)
	}

	updateDynamic := func(pod *v1.Pod) (*v1.Pod, error) {
		body, marshalErr := json.Marshal(pod)
		if marshalErr != nil {
			return nil, marshalErr
		}
		result := &v1.Pod{}
		reqErr := client.CoreV1().RESTClient().Put().
			Namespace(ns.Name).
			Resource("pods").
			Name(pod.Name).
			SubResource("dynamic").
			Body(body).
			Do(ctx).
			Into(result)
		return result, reqErr
	}

	// Wait until NodeDeclaredFeatures informer observes node before testing admission rejections.
	var afterAllowed *v1.Pod
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		attempt := created.DeepCopy()
		attempt.Spec.Volumes = append(attempt.Spec.Volumes,
			v1.Volume{
				Name: "ok-sec",
				VolumeSource: v1.VolumeSource{
					Secret: &v1.SecretVolumeSource{SecretName: "allowed-secret"},
				},
			},
			v1.Volume{
				Name: "dyn-img",
				VolumeSource: v1.VolumeSource{
					Image: &v1.ImageVolumeSource{
						Reference:  "registry.k8s.io/pause:3.10",
						PullPolicy: v1.PullIfNotPresent,
					},
				},
			},
		)
		attempt.Spec.Containers = append(attempt.Spec.Containers, v1.Container{
			Name:  "c2",
			Image: "busybox:1.36",
			VolumeMounts: []v1.VolumeMount{
				{Name: "ok-sec", MountPath: "/mnt/ok-sec"},
				{Name: "dyn-img", MountPath: "/mnt/dyn-img"},
			},
		})
		var updateErr error
		afterAllowed, updateErr = updateDynamic(attempt)
		return updateErr == nil, nil
	})
	if err != nil {
		t.Fatalf("expected allowed secret and image volume via /dynamic to succeed, got: %v", err)
	}

	// AlwaysPullImages must mutate newly added Image volume PullPolicy to Always.
	var foundImgVol *v1.ImageVolumeSource
	for i := range afterAllowed.Spec.Volumes {
		if afterAllowed.Spec.Volumes[i].Name == "dyn-img" {
			foundImgVol = afterAllowed.Spec.Volumes[i].Image
		}
	}
	if foundImgVol == nil || foundImgVol.PullPolicy != v1.PullAlways {
		t.Fatalf("expected AlwaysPullImages to set Image.PullPolicy=Always on /dynamic, got: %+v", foundImgVol)
	}

	// ServiceAccount enforce-mountable-secrets must reject adding an unpermitted Secret volume via /dynamic.
	forbiddenAttempt := afterAllowed.DeepCopy()
	forbiddenAttempt.Spec.Volumes = append(forbiddenAttempt.Spec.Volumes, v1.Volume{
		Name: "bad-sec",
		VolumeSource: v1.VolumeSource{
			Secret: &v1.SecretVolumeSource{SecretName: "unlisted-secret"},
		},
	})
	forbiddenAttempt.Spec.Containers = append(forbiddenAttempt.Spec.Containers, v1.Container{
		Name:         "c-bad-sec",
		Image:        "busybox:1.36",
		VolumeMounts: []v1.VolumeMount{{Name: "bad-sec", MountPath: "/mnt/bad-sec"}},
	})
	if _, err := updateDynamic(forbiddenAttempt); !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "does not reference") || !strings.Contains(err.Error(), "unlisted-secret") {
		t.Fatalf("expected Forbidden error for unlisted-secret from ServiceAccount enforce-mountable-secrets on /dynamic, got: %v", err)
	}

	// NodeDeclaredFeatureValidator must reject introducing bindMountOptions via /dynamic when the node does not declare VolumeBindMountOptions.
	bindOptsAttempt := afterAllowed.DeepCopy()
	bindOptsAttempt.Spec.Containers = append(bindOptsAttempt.Spec.Containers, v1.Container{
		Name:  "c-bind-opts",
		Image: "busybox:1.36",
		VolumeMounts: []v1.VolumeMount{{
			Name:             "ok-sec",
			MountPath:        "/mnt/ok-sec-nosuid",
			BindMountOptions: []string{"nosuid"},
		}},
	})
	if _, err := updateDynamic(bindOptsAttempt); !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "VolumeBindMountOptions") {
		t.Fatalf("expected Forbidden error from NodeDeclaredFeatureValidator for VolumeBindMountOptions on /dynamic, got: %v", err)
	}

	// PodSecurity must treat adding volumes via /dynamic as significant updates.
	psaNS := framework.CreateNamespaceOrDie(client, "pod-dyn-psa", t)
	defer framework.DeleteNamespaceOrDie(client, psaNS, t)
	if _, err := client.CoreV1().ServiceAccounts(psaNS.Name).Create(ctx, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: psaNS.Name},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create default ServiceAccount in %s: %v", psaNS.Name, err)
	}

	privPod, err := client.CoreV1().Pods(psaNS.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "psa-pod",
			Namespace: psaNS.Name,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Volumes: []v1.Volume{
				{Name: "vol-keep", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
				{Name: "vol-remove", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
			},
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "busybox:1.36",
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod before tightening namespace PSA label: %v", err)
	}
	privStatus := privPod.DeepCopy()
	privStatus.Status.Phase = v1.PodRunning
	privStatus.Status.Conditions = []v1.PodCondition{{Type: v1.PodInitialized, Status: v1.ConditionTrue}}
	privPod, err = client.CoreV1().Pods(psaNS.Name).UpdateStatus(ctx, privStatus, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to mark pod running: %v", err)
	}

	psaNSLabeled := psaNS.DeepCopy()
	psaNSLabeled.Labels = map[string]string{"pod-security.kubernetes.io/enforce": "restricted"}
	if _, err := client.CoreV1().Namespaces().Update(ctx, psaNSLabeled, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("failed to tighten PSA enforce label on namespace: %v", err)
	}

	addVolAttempt := privPod.DeepCopy()
	addVolAttempt.Spec.Volumes = append(addVolAttempt.Spec.Volumes, v1.Volume{
		Name:         "vol-added",
		VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
	})
	err = wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		_, psaErr := client.CoreV1().Pods(psaNS.Name).UpdateDynamic(ctx, addVolAttempt.Name, addVolAttempt, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
		return apierrors.IsForbidden(psaErr) && strings.Contains(psaErr.Error(), "violates PodSecurity"), nil
	})
	if err != nil {
		t.Fatalf("expected PodSecurity namespace informer to sync restricted label: %v", err)
	}
	if _, psaErr := client.CoreV1().Pods(psaNS.Name).UpdateDynamic(ctx, addVolAttempt.Name, addVolAttempt, metav1.UpdateOptions{}); !apierrors.IsForbidden(psaErr) || !strings.Contains(psaErr.Error(), "violates PodSecurity") {
		t.Fatalf("expected PodSecurity to forbid adding volume via /dynamic on non-compliant pod in restricted namespace, got: %v", psaErr)
	}

	// Removing an unreferenced volume remains an insignificant update and is permitted.
	removeVolAttempt := privPod.DeepCopy()
	var keptVols []v1.Volume
	for _, v := range removeVolAttempt.Spec.Volumes {
		if v.Name != "vol-remove" {
			keptVols = append(keptVols, v)
		}
	}
	removeVolAttempt.Spec.Volumes = keptVols
	if _, err := client.CoreV1().Pods(psaNS.Name).UpdateDynamic(ctx, removeVolAttempt.Name, removeVolAttempt, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("expected removing volume via /dynamic to be permitted as insignificant by PodSecurity, got: %v", err)
	}
}
