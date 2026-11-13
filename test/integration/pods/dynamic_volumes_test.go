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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-dynamic-volumes", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx := context.Background()
	nodeName := "dynamic-vol-node"
	restartPolicy := &v1.ContainerDynamicVolumeMountPolicy{
		RestartPolicy: v1.DynamicVolumeMountRestartContainer,
	}

	createNode := func(t *testing.T, declaredFeatures []string) {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
			},
			Status: v1.NodeStatus{
				NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
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
						Name:                     "c1",
						Image:                    "busybox",
						DynamicVolumeMountPolicy: restartPolicy,
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

	t.Run("rejects dynamicVolumeMountPolicy RestartContainer when pod restartPolicy is Never", func(t *testing.T) {
		neverPod := basePod()
		neverPod.Spec.RestartPolicy = v1.RestartPolicyNever
		if _, err := client.CoreV1().Pods(ns.Name).Create(ctx, neverPod, metav1.CreateOptions{}); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error creating pod with restartPolicy=Never and DynamicVolumeMountRestartContainer, got: %v", err)
		}
	})

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
		// Pod creation does not seed VolumeStatuses so older kubelets never leave static volumes stuck in Pending.
		if len(created.Status.VolumeStatuses) != 0 {
			t.Fatalf("expected empty VolumeStatuses on pod create, got: %+v", created.Status.VolumeStatuses)
		}
		created = markPodRunningAndInitialized(t, created)

		// Rejects mutating DynamicVolumeMountPolicy on an existing container via /dynamic.
		mutatedPolicy := created.DeepCopy()
		mutatedPolicy.Spec.Containers[0].DynamicVolumeMountPolicy = nil
		if _, err := updateDynamic(ctx, mutatedPolicy); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when mutating DynamicVolumeMountPolicy on existing container, got: %v", err)
		}

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

		// Rejects removing a mounted volume or volumeMount before VolumeStatuses is populated.
		prematureRemove := afterUnreferenced.DeepCopy()
		prematureRemove.Spec.Volumes = prematureRemove.Spec.Volumes[1:]
		prematureRemove.Spec.Containers[0].VolumeMounts = nil
		if _, err := updateDynamic(ctx, prematureRemove); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when removing mounted volume before VolumeStatuses is populated, got: %v", err)
		}

		var afterAdd *v1.Pod
		updated := afterUnreferenced.DeepCopy()
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
		afterAdd, err = updateDynamic(ctx, updated)
		if err != nil {
			t.Fatalf("unexpected error adding configMap volume via /dynamic: %v", err)
		}
		if len(afterAdd.Spec.Volumes) != 4 {
			t.Fatalf("expected 4 volumes after /dynamic update, got %d", len(afterAdd.Spec.Volumes))
		}
		if len(afterAdd.Status.VolumeStatuses) != 1 ||
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

		// Rejects adding a container via /dynamic when DynamicContainers feature gate is disabled.
		withAddedContainer := afterAdd.DeepCopy()
		withAddedContainer.Spec.Containers = append(withAddedContainer.Spec.Containers, v1.Container{
			Name:  "c2",
			Image: "busybox",
		})
		if _, err := updateDynamic(ctx, withAddedContainer); !apierrors.IsInvalid(err) {
			t.Fatalf("expected Invalid error when adding container via /dynamic with DynamicContainers disabled, got: %v", err)
		}

		// Verify PodStatus.VolumeStatuses supports both HostPhase and CSIVolumeHealth HealthStatus co-existing via /status.
		withStatus := afterAdd.DeepCopy()
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
		if len(statusUpdated.Status.VolumeStatuses) != 3 {
			t.Fatalf("expected 3 VolumeStatuses in PodStatus, got %+v", statusUpdated.Status.VolumeStatuses)
		}

		// Remove vol-cm via /dynamic before kubelet mounts it (vol-cm is still Pending in VolumeStatuses).
		afterRemove := statusUpdated.DeepCopy()
		afterRemove.Spec.Volumes = afterRemove.Spec.Volumes[:3]
		afterRemove.Spec.Containers[0].VolumeMounts = afterRemove.Spec.Containers[0].VolumeMounts[:1]
		removedPod, err := updateDynamic(ctx, afterRemove)
		if err != nil {
			t.Fatalf("unexpected error removing volume via /dynamic: %v", err)
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
		rapidReAdd.Spec.Containers[0].VolumeMounts = append(rapidReAdd.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      "vol-cm",
			MountPath: "/mnt/cm",
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
		reAddAfterClear.Spec.Containers[0].VolumeMounts = append(reAddAfterClear.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      "vol-cm",
			MountPath: "/mnt/cm",
		})
		afterReAdd, err := updateDynamic(ctx, reAddAfterClear)
		if err != nil {
			t.Fatalf("expected re-adding vol-cm to succeed once cleared at current generation, got: %v", err)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeBindMountOptions, true)
	flags := []string{
		"--endpoint-reconciler-type=none",
		"--enable-admission-plugins=ServiceAccount,AlwaysPullImages",
	}
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-dynamic-vol-admission", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx := context.Background()
	nodeName := "dyn-adm-node"
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		Status: v1.NodeStatus{
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
			DeclaredFeatures: []string{dynamicvolumesfeature.Feature.Name()},
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
					DynamicVolumeMountPolicy: &v1.ContainerDynamicVolumeMountPolicy{
						RestartPolicy: v1.DynamicVolumeMountRestartContainer,
					},
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
		attempt.Spec.Containers[0].VolumeMounts = append(attempt.Spec.Containers[0].VolumeMounts,
			v1.VolumeMount{Name: "ok-sec", MountPath: "/mnt/ok-sec"},
			v1.VolumeMount{Name: "dyn-img", MountPath: "/mnt/dyn-img"},
		)
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
	forbiddenAttempt.Spec.Containers[0].VolumeMounts = append(forbiddenAttempt.Spec.Containers[0].VolumeMounts,
		v1.VolumeMount{Name: "bad-sec", MountPath: "/mnt/bad-sec"},
	)
	if _, err := updateDynamic(forbiddenAttempt); !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "does not reference") || !strings.Contains(err.Error(), "unlisted-secret") {
		t.Fatalf("expected Forbidden error for unlisted-secret from ServiceAccount enforce-mountable-secrets on /dynamic, got: %v", err)
	}

	// NodeDeclaredFeatureValidator must reject introducing bindMountOptions via /dynamic when the node does not declare VolumeBindMountOptions.
	bindOptsAttempt := afterAllowed.DeepCopy()
	bindOptsAttempt.Spec.Containers[0].VolumeMounts = append(bindOptsAttempt.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:             "ok-sec",
		MountPath:        "/mnt/ok-sec-nosuid",
		BindMountOptions: []string{"nosuid"},
	})
	if _, err := updateDynamic(bindOptsAttempt); !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "VolumeBindMountOptions") {
		t.Fatalf("expected Forbidden error from NodeDeclaredFeatureValidator for VolumeBindMountOptions on /dynamic, got: %v", err)
	}
}
