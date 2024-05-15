/*
Copyright 2018 The Kubernetes Authors.

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

package statefulset

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/statefulset"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const (
	interval = 100 * time.Millisecond
	timeout  = 60 * time.Second
)

// TestVolumeTemplateNoopUpdate ensures embedded StatefulSet objects with embedded PersistentVolumes can be updated
func TestVolumeTemplateNoopUpdate(t *testing.T) {
	// Start the server with default storage setup
	server := apiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	c, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// Use an unstructured client to ensure we send exactly the bytes we expect for the embedded PVC template
	sts := &unstructured.Unstructured{}
	err = json.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "StatefulSet",
		"metadata": {"name": "web"},
		"spec": {
		  "selector": {"matchLabels": {"app": "nginx"}},
		  "serviceName": "nginx",
		  "replicas": 3,
		  "template": {
			"metadata": {"labels": {"app": "nginx"}},
			"spec": {
			  "terminationGracePeriodSeconds": 10,
			  "containers": [{
				  "name": "nginx",
				  "image": "registry.k8s.io/nginx-slim:0.8",
				  "ports": [{"containerPort": 80,"name": "web"}],
				  "volumeMounts": [{"name": "www","mountPath": "/usr/share/nginx/html"}]
			  }]
			}
		  },
		  "volumeClaimTemplates": [{
			  "apiVersion": "v1",
			  "kind": "PersistentVolumeClaim",
			  "metadata": {"name": "www"},
			  "spec": {
				"accessModes": ["ReadWriteOnce"],
				"storageClassName": "my-storage-class",
				"resources": {"requests": {"storage": "1Gi"}}
			  }
			}
		  ]
		}
	  }`), &sts.Object)
	if err != nil {
		t.Fatal(err)
	}

	stsClient := c.Resource(appsv1.SchemeGroupVersion.WithResource("statefulsets")).Namespace("default")

	// Create the statefulset
	persistedSTS, err := stsClient.Create(context.TODO(), sts, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Update with the original spec (all the same defaulting should apply, should be a no-op and pass validation
	originalSpec, ok, err := unstructured.NestedFieldCopy(sts.Object, "spec")
	if err != nil || !ok {
		t.Fatal(err, ok)
	}
	err = unstructured.SetNestedField(persistedSTS.Object, originalSpec, "spec")
	if err != nil {
		t.Fatal(err)
	}
	_, err = stsClient.Update(context.TODO(), persistedSTS, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestSpecReplicasChange(t *testing.T) {
	tCtx, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-spec-replicas-change", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	cancel := runControllerAndInformers(tCtx, rm, informers)
	defer cancel()

	createHeadlessService(t, c, newHeadlessService(ns.Name))
	sts := newSTS("sts", ns.Name, 2)
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Update .Spec.Replicas and verify .Status.Replicas is changed accordingly
	scaleSTS(t, c, sts, 3)
	scaleSTS(t, c, sts, 0)
	scaleSTS(t, c, sts, 2)

	// Add a template annotation change to test STS's status does update
	// without .Spec.Replicas change
	stsClient := c.AppsV1().StatefulSets(ns.Name)
	var oldGeneration int64
	newSTS := updateSTS(t, stsClient, sts.Name, func(sts *appsv1.StatefulSet) {
		oldGeneration = sts.Generation
		sts.Spec.Template.Annotations = map[string]string{"test": "annotation"}
	})
	savedGeneration := newSTS.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("failed to verify .Generation has incremented for sts %s", sts.Name)
	}

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newSTS, err := stsClient.Get(context.TODO(), sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newSTS.Status.ObservedGeneration >= savedGeneration, nil
	}); err != nil {
		t.Fatalf("failed to verify .Status.ObservedGeneration has incremented for sts %s: %v", sts.Name, err)
	}
}

func TestDeletingAndTerminatingPods(t *testing.T) {
	tCtx, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-deleting-and-failed-pods", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	cancel := runControllerAndInformers(tCtx, rm, informers)
	defer cancel()

	podCount := 3

	labelMap := labelMap()
	sts := newSTS("sts", ns.Name, podCount)
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Verify STS creates 3 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap)
	if len(pods.Items) != podCount {
		t.Fatalf("len(pods) = %d, want %d", len(pods.Items), podCount)
	}

	// Set first pod as deleting pod
	// Set finalizers for the pod to simulate pending deletion status
	deletingPod := &pods.Items[0]
	updatePod(t, podClient, deletingPod.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{"fake.example.com/blockDeletion"}
	})
	if err := c.CoreV1().Pods(ns.Name).Delete(context.TODO(), deletingPod.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting pod %s: %v", deletingPod.Name, err)
	}

	// Set second pod as failed pod
	failedPod := &pods.Items[1]
	updatePodStatus(t, podClient, failedPod.Name, func(pod *v1.Pod) {
		pod.Status.Phase = v1.PodFailed
	})

	// Set third pod as succeeded pod
	succeededPod := &pods.Items[2]
	updatePodStatus(t, podClient, succeededPod.Name, func(pod *v1.Pod) {
		pod.Status.Phase = v1.PodSucceeded
	})

	exists := func(pods []v1.Pod, uid types.UID) bool {
		for _, pod := range pods {
			if pod.UID == uid {
				return true
			}
		}
		return false
	}

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		// Verify only 3 pods exist: deleting pod and new pod replacing failed pod
		pods = getPods(t, podClient, labelMap)
		if len(pods.Items) != podCount {
			return false, nil
		}

		// Verify deleting pod still exists
		// Immediately return false with an error if it does not exist
		if !exists(pods.Items, deletingPod.UID) {
			return false, fmt.Errorf("expected deleting pod %s still exists, but it is not found", deletingPod.Name)
		}
		// Verify failed pod does not exist anymore
		if exists(pods.Items, failedPod.UID) {
			return false, nil
		}
		// Verify succeeded pod does not exist anymore
		if exists(pods.Items, succeededPod.UID) {
			return false, nil
		}
		// Verify all pods have non-terminated status
		for _, pod := range pods.Items {
			if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to verify failed pod %s has been replaced with a new non-failed pod, and deleting pod %s survives: %v", failedPod.Name, deletingPod.Name, err)
	}

	// Remove finalizers of deleting pod to simulate successful deletion
	updatePod(t, podClient, deletingPod.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{}
	})

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		// Verify only 2 pods exist: new non-deleting pod replacing deleting pod and the non-failed pod
		pods = getPods(t, podClient, labelMap)
		if len(pods.Items) != podCount {
			return false, nil
		}
		// Verify deleting pod does not exist anymore
		return !exists(pods.Items, deletingPod.UID), nil
	}); err != nil {
		t.Fatalf("failed to verify deleting pod %s has been replaced with a new non-deleting pod: %v", deletingPod.Name, err)
	}
}

func TestStatefulSetAvailable(t *testing.T) {
	tests := []struct {
		name           string
		totalReplicas  int32
		readyReplicas  int32
		activeReplicas int32
	}{
		{
			name:           "only certain replicas would become active",
			totalReplicas:  4,
			readyReplicas:  3,
			activeReplicas: 2,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx, closeFn, rm, informers, c := scSetup(t)
			defer closeFn()
			ns := framework.CreateNamespaceOrDie(c, "test-available-pods", t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)
			cancel := runControllerAndInformers(tCtx, rm, informers)
			defer cancel()

			labelMap := labelMap()
			sts := newSTS("sts", ns.Name, 4)
			sts.Spec.MinReadySeconds = int32(3600)
			stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
			sts = stss[0]
			waitSTSStable(t, c, sts)

			// Verify STS creates 4 pods
			podClient := c.CoreV1().Pods(ns.Name)
			pods := getPods(t, podClient, labelMap)
			if len(pods.Items) != 4 {
				t.Fatalf("len(pods) = %d, want 4", len(pods.Items))
			}

			// Separate 3 pods into their own list
			firstPodList := &v1.PodList{Items: pods.Items[:1]}
			secondPodList := &v1.PodList{Items: pods.Items[1:2]}
			thirdPodList := &v1.PodList{Items: pods.Items[2:]}
			// First pod: Running, but not Ready
			// by setting the Ready condition to false with LastTransitionTime to be now
			setPodsReadyCondition(t, c, firstPodList, v1.ConditionFalse, time.Now())
			// Second pod: Running and Ready, but not Available
			// by setting LastTransitionTime to now
			setPodsReadyCondition(t, c, secondPodList, v1.ConditionTrue, time.Now())
			// Third pod: Running, Ready, and Available
			// by setting LastTransitionTime to more than 3600 seconds ago
			setPodsReadyCondition(t, c, thirdPodList, v1.ConditionTrue, time.Now().Add(-120*time.Minute))

			stsClient := c.AppsV1().StatefulSets(ns.Name)
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				newSts, err := stsClient.Get(context.TODO(), sts.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				// Verify 4 pods exist, 3 pods are Ready, and 2 pods are Available
				return newSts.Status.Replicas == test.totalReplicas && newSts.Status.ReadyReplicas == test.readyReplicas && newSts.Status.AvailableReplicas == test.activeReplicas, nil
			}); err != nil {
				t.Fatalf("Failed to verify number of Replicas, ReadyReplicas and AvailableReplicas of rs %s to be as expected: %v", sts.Name, err)
			}
		})
	}
}

func setPodsReadyCondition(t *testing.T, clientSet clientset.Interface, pods *v1.PodList, conditionStatus v1.ConditionStatus, lastTransitionTime time.Time) {
	replicas := int32(len(pods.Items))
	var readyPods int32
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		readyPods = 0
		for i := range pods.Items {
			pod := &pods.Items[i]
			if podutil.IsPodReady(pod) {
				readyPods++
				continue
			}
			pod.Status.Phase = v1.PodRunning
			_, condition := podutil.GetPodCondition(&pod.Status, v1.PodReady)
			if condition != nil {
				condition.Status = conditionStatus
				condition.LastTransitionTime = metav1.Time{Time: lastTransitionTime}
			} else {
				condition = &v1.PodCondition{
					Type:               v1.PodReady,
					Status:             conditionStatus,
					LastTransitionTime: metav1.Time{Time: lastTransitionTime},
				}
				pod.Status.Conditions = append(pod.Status.Conditions, *condition)
			}
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{})
			if err != nil {
				// When status fails to be updated, we continue to next pod
				continue
			}
			readyPods++
		}
		return readyPods >= replicas, nil
	})
	if err != nil {
		t.Fatalf("failed to mark all StatefulSet pods to ready: %v", err)
	}
}

// add for issue: https://github.com/kubernetes/kubernetes/issues/108837
func TestStatefulSetStatusWithPodFail(t *testing.T) {
	tCtx := ktesting.Init(t)
	limitedPodNumber := 2
	c, config, closeFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerConfig: func(config *controlplane.Config) {
			config.ControlPlane.Generic.AdmissionControl = &fakePodFailAdmission{
				limitedPodNumber: limitedPodNumber,
			}
		},
	})
	defer closeFn()
	defer tCtx.Cancel("test has completed")
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "statefulset-informers")), resyncPeriod)
	ssc := statefulset.NewStatefulSetController(
		tCtx,
		informers.Core().V1().Pods(),
		informers.Apps().V1().StatefulSets(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Apps().V1().ControllerRevisions(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "statefulset-controller")),
	)

	ns := framework.CreateNamespaceOrDie(c, "test-pod-fail", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	informers.Start(tCtx.Done())
	go ssc.Run(tCtx, 5)

	sts := newSTS("sts", ns.Name, 4)
	_, err := c.AppsV1().StatefulSets(sts.Namespace).Create(tCtx, sts, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Could not create statefulSet %s: %v", sts.Name, err)
	}

	wantReplicas := limitedPodNumber
	var gotReplicas int32
	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newSTS, err := c.AppsV1().StatefulSets(sts.Namespace).Get(tCtx, sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		gotReplicas = newSTS.Status.Replicas
		return gotReplicas == int32(wantReplicas), nil
	}); err != nil {
		t.Fatalf("StatefulSet %s status has %d replicas, want replicas %d: %v", sts.Name, gotReplicas, wantReplicas, err)
	}
}

func TestAutodeleteOwnerRefs(t *testing.T) {
	tests := []struct {
		namespace         string
		policy            appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy
		expectPodOwnerRef bool
		expectSetOwnerRef bool
	}{
		{
			namespace: "always-retain",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: false,
			expectSetOwnerRef: false,
		},
		{
			namespace: "delete-on-scaledown-only",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: true,
			expectSetOwnerRef: false,
		},
		{
			namespace: "delete-with-set-only",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: false,
			expectSetOwnerRef: true,
		},
		{
			namespace: "always-delete",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: true,
			expectSetOwnerRef: true,
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)

	tCtx, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	cancel := runControllerAndInformers(tCtx, rm, informers)
	defer cancel()

	for _, test := range tests {
		t.Run(test.namespace, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(c, test.namespace, t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)

			sts := newSTS("sts", ns.Name, 3)
			sts.Spec.PersistentVolumeClaimRetentionPolicy = &test.policy
			stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
			sts = stss[0]
			waitSTSStable(t, c, sts)

			// Verify StatefulSet ownerref has been added as appropriate.
			pvcClient := c.CoreV1().PersistentVolumeClaims(ns.Name)
			pvcs := getStatefulSetPVCs(t, pvcClient, sts)
			for _, pvc := range pvcs {
				verifyOwnerRef(t, pvc, "StatefulSet", test.expectSetOwnerRef)
				verifyOwnerRef(t, pvc, "Pod", false)
			}

			// Scale down to 1 pod and verify Pod ownerrefs as appropriate.
			one := int32(1)
			sts.Spec.Replicas = &one
			waitSTSStable(t, c, sts)

			pvcs = getStatefulSetPVCs(t, pvcClient, sts)
			for i, pvc := range pvcs {
				verifyOwnerRef(t, pvc, "StatefulSet", test.expectSetOwnerRef)
				if i == 0 {
					verifyOwnerRef(t, pvc, "Pod", false)
				} else {
					verifyOwnerRef(t, pvc, "Pod", test.expectPodOwnerRef)
				}
			}
		})
	}
}

func TestDeletingPodForRollingUpdatePartition(t *testing.T) {
	tCtx, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-deleting-pod-for-rolling-update-partition", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	cancel := runControllerAndInformers(tCtx, rm, informers)
	defer cancel()

	labelMap := labelMap()
	sts := newSTS("sts", ns.Name, 2)
	sts.Spec.UpdateStrategy = appsv1.StatefulSetUpdateStrategy{
		Type: appsv1.RollingUpdateStatefulSetStrategyType,
		RollingUpdate: func() *appsv1.RollingUpdateStatefulSetStrategy {
			return &appsv1.RollingUpdateStatefulSetStrategy{
				Partition: ptr.To[int32](1),
			}
		}(),
	}
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Verify STS creates 2 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap)
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
	}
	// Setting all pods in Running, Ready, and Available
	setPodsReadyCondition(t, c, &v1.PodList{Items: pods.Items}, v1.ConditionTrue, time.Now())

	// 1. Roll out a new image.
	oldImage := sts.Spec.Template.Spec.Containers[0].Image
	newImage := "new-image"
	if oldImage == newImage {
		t.Fatalf("bad test setup, statefulSet %s roll out with the same image", sts.Name)
	}
	// Set finalizers for the pod-0 to trigger pod recreation failure while the status UpdateRevision is bumped
	pod0 := &pods.Items[0]
	updatePod(t, podClient, pod0.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{"fake.example.com/blockDeletion"}
	})

	stsClient := c.AppsV1().StatefulSets(ns.Name)
	_ = updateSTS(t, stsClient, sts.Name, func(sts *appsv1.StatefulSet) {
		sts.Spec.Template.Spec.Containers[0].Image = newImage
	})

	// Await for the pod-1 to be recreated, while pod-0 remains running
	if err := wait.PollUntilContextTimeout(tCtx, pollInterval, pollTimeout, false, func(ctx context.Context) (bool, error) {
		ss, err := stsClient.Get(ctx, sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		pods := getPods(t, podClient, labelMap)
		recreatedPods := v1.PodList{}
		for _, pod := range pods.Items {
			if pod.Status.Phase == v1.PodPending {
				recreatedPods.Items = append(recreatedPods.Items, pod)
			}
		}
		setPodsReadyCondition(t, c, &v1.PodList{Items: recreatedPods.Items}, v1.ConditionTrue, time.Now())
		return ss.Status.UpdatedReplicas == *ss.Spec.Replicas-*sts.Spec.UpdateStrategy.RollingUpdate.Partition && ss.Status.Replicas == *ss.Spec.Replicas && ss.Status.ReadyReplicas == *ss.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("failed to await for pod-1 to be recreated by sts %s: %v", sts.Name, err)
	}

	// Mark pod-0 as terminal and not ready
	updatePodStatus(t, podClient, pod0.Name, func(pod *v1.Pod) {
		pod.Status.Phase = v1.PodFailed
	})

	// Make sure pod-0 gets deletion timestamp so that it is recreated
	if err := c.CoreV1().Pods(ns.Name).Delete(context.TODO(), pod0.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting pod %s: %v", pod0.Name, err)
	}

	// Await for pod-0 to be not ready
	if err := wait.PollUntilContextTimeout(tCtx, pollInterval, pollTimeout, false, func(ctx context.Context) (bool, error) {
		ss, err := stsClient.Get(ctx, sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ss.Status.ReadyReplicas == *ss.Spec.Replicas-1, nil
	}); err != nil {
		t.Fatalf("failed to await for pod-0 to be not counted as ready in status of sts %s: %v", sts.Name, err)
	}

	// Remove the finalizer to allow recreation
	updatePod(t, podClient, pod0.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{}
	})

	// Await for pod-0 to be recreated and make it running
	if err := wait.PollUntilContextTimeout(tCtx, pollInterval, pollTimeout, false, func(ctx context.Context) (bool, error) {
		pods := getPods(t, podClient, labelMap)
		recreatedPods := v1.PodList{}
		for _, pod := range pods.Items {
			if pod.Status.Phase == v1.PodPending {
				recreatedPods.Items = append(recreatedPods.Items, pod)
			}
		}
		setPodsReadyCondition(t, c, &v1.PodList{Items: recreatedPods.Items}, v1.ConditionTrue, time.Now().Add(-120*time.Minute))
		return len(recreatedPods.Items) > 0, nil
	}); err != nil {
		t.Fatalf("failed to await for pod-0 to be recreated by sts %s: %v", sts.Name, err)
	}

	// Await for all stateful set status to record all replicas as ready
	if err := wait.PollUntilContextTimeout(tCtx, pollInterval, pollTimeout, false, func(ctx context.Context) (bool, error) {
		ss, err := stsClient.Get(ctx, sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return ss.Status.ReadyReplicas == *ss.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("failed to verify .Spec.Template.Spec.Containers[0].Image is updated for sts %s: %v", sts.Name, err)
	}

	// Verify 3 pods exist
	pods = getPods(t, podClient, labelMap)
	if len(pods.Items) != int(*sts.Spec.Replicas) {
		t.Fatalf("Unexpected number of pods")
	}

	// Verify pod images
	for i := range pods.Items {
		if i < int(*sts.Spec.UpdateStrategy.RollingUpdate.Partition) {
			if pods.Items[i].Spec.Containers[0].Image != oldImage {
				t.Fatalf("Pod %s has image %s not equal to old image %s", pods.Items[i].Name, pods.Items[i].Spec.Containers[0].Image, oldImage)
			}
		} else {
			if pods.Items[i].Spec.Containers[0].Image != newImage {
				t.Fatalf("Pod %s has image %s not equal to new image %s", pods.Items[i].Name, pods.Items[i].Spec.Containers[0].Image, newImage)
			}
		}
	}
}

func TestStatefulSetStartOrdinal(t *testing.T) {
	tests := []struct {
		ordinals         *appsv1.StatefulSetOrdinals
		name             string
		namespace        string
		replicas         int
		expectedPodNames []string
	}{
		{
			name:             "default start ordinal, no ordinals set",
			namespace:        "no-ordinals",
			replicas:         3,
			expectedPodNames: []string{"sts-0", "sts-1", "sts-2"},
		},
		{
			name:             "default start ordinal",
			namespace:        "no-start-ordinals",
			ordinals:         &appsv1.StatefulSetOrdinals{},
			replicas:         3,
			expectedPodNames: []string{"sts-0", "sts-1", "sts-2"},
		},
		{
			name:      "start ordinal 4",
			namespace: "start-ordinal-4",
			ordinals: &appsv1.StatefulSetOrdinals{
				Start: 4,
			},
			replicas:         4,
			expectedPodNames: []string{"sts-4", "sts-5", "sts-6", "sts-7"},
		},
		{
			name:      "start ordinal 5",
			namespace: "start-ordinal-5",
			ordinals: &appsv1.StatefulSetOrdinals{
				Start: 2,
			},
			replicas:         7,
			expectedPodNames: []string{"sts-2", "sts-3", "sts-4", "sts-5", "sts-6", "sts-7", "sts-8"},
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetStartOrdinal, true)
	tCtx, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	cancel := runControllerAndInformers(tCtx, rm, informers)
	defer cancel()

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(c, test.namespace, t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)

			// Label map is the map of pod labels used in newSTS()
			labelMap := labelMap()
			sts := newSTS("sts", ns.Name, test.replicas)
			sts.Spec.Ordinals = test.ordinals
			stss := createSTSs(t, c, []*appsv1.StatefulSet{sts})
			sts = stss[0]
			waitSTSStable(t, c, sts)

			podClient := c.CoreV1().Pods(ns.Name)
			pods := getPods(t, podClient, labelMap)
			if len(pods.Items) != test.replicas {
				t.Errorf("len(pods) = %v, want %v", len(pods.Items), test.replicas)
			}

			var podNames []string
			for _, pod := range pods.Items {
				podNames = append(podNames, pod.Name)
			}
			ignoreOrder := cmpopts.SortSlices(func(a, b string) bool {
				return a < b
			})

			// Validate all the expected pods were created.
			if diff := cmp.Diff(test.expectedPodNames, podNames, ignoreOrder); diff != "" {
				t.Errorf("Unexpected pod names: (-want +got): %v", diff)
			}

			// Scale down to 1 pod and verify it matches the first pod.
			scaleSTS(t, c, sts, 1)
			waitSTSStable(t, c, sts)

			pods = getPods(t, podClient, labelMap)
			if len(pods.Items) != 1 {
				t.Errorf("len(pods) = %v, want %v", len(pods.Items), 1)
			}
			if pods.Items[0].Name != test.expectedPodNames[0] {
				t.Errorf("Unexpected singleton pod name: got = %v, want %v", pods.Items[0].Name, test.expectedPodNames[0])
			}
		})
	}
}
