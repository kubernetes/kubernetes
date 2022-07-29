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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
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
	closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-spec-replicas-change", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	cancel := runControllerAndInformers(rm, informers)
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

func TestDeletingAndFailedPods(t *testing.T) {
	closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-deleting-and-failed-pods", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	cancel := runControllerAndInformers(rm, informers)
	defer cancel()

	labelMap := labelMap()
	sts := newSTS("sts", ns.Name, 2)
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Verify STS creates 2 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap)
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
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

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		// Verify only 2 pods exist: deleting pod and new pod replacing failed pod
		pods = getPods(t, podClient, labelMap)
		if len(pods.Items) != 2 {
			return false, nil
		}
		// Verify deleting pod still exists
		// Immediately return false with an error if it does not exist
		if pods.Items[0].UID != deletingPod.UID && pods.Items[1].UID != deletingPod.UID {
			return false, fmt.Errorf("expected deleting pod %s still exists, but it is not found", deletingPod.Name)
		}
		// Verify failed pod does not exist anymore
		if pods.Items[0].UID == failedPod.UID || pods.Items[1].UID == failedPod.UID {
			return false, nil
		}
		// Verify both pods have non-failed status
		return pods.Items[0].Status.Phase != v1.PodFailed && pods.Items[1].Status.Phase != v1.PodFailed, nil
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
		if len(pods.Items) != 2 {
			return false, nil
		}
		// Verify deleting pod does not exist anymore
		return pods.Items[0].UID != deletingPod.UID && pods.Items[1].UID != deletingPod.UID, nil
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
			closeFn, rm, informers, c := scSetup(t)
			defer closeFn()
			ns := framework.CreateNamespaceOrDie(c, "test-available-pods", t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)
			cancel := runControllerAndInformers(rm, informers)
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
	limitedPodNumber := 2
	c, config, closeFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerConfig: func(config *controlplane.Config) {
			config.GenericConfig.AdmissionControl = &fakePodFailAdmission{
				limitedPodNumber: limitedPodNumber,
			}
		},
	})
	defer closeFn()

	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "statefulset-informers")), resyncPeriod)

	ssc := statefulset.NewStatefulSetController(
		informers.Core().V1().Pods(),
		informers.Apps().V1().StatefulSets(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Apps().V1().ControllerRevisions(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "statefulset-controller")),
	)

	ns := framework.CreateNamespaceOrDie(c, "test-pod-fail", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	informers.Start(ctx.Done())
	go ssc.Run(ctx, 5)

	sts := newSTS("sts", ns.Name, 4)
	_, err := c.AppsV1().StatefulSets(sts.Namespace).Create(context.TODO(), sts, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Could not create statefuleSet %s: %v", sts.Name, err)
	}

	wantReplicas := limitedPodNumber
	var gotReplicas int32
	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newSTS, err := c.AppsV1().StatefulSets(sts.Namespace).Get(context.TODO(), sts.Name, metav1.GetOptions{})
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
		name              string
		policy            appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy
		expectPodOwnerRef bool
		expectSetOwnerRef bool
	}{
		{
			name: "always retain",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: false,
			expectSetOwnerRef: false,
		},
		{
			name: "delete on scaledown only",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: true,
			expectSetOwnerRef: false,
		},
		{
			name: "delete with set only",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: false,
			expectSetOwnerRef: true,
		},
		{
			name: "always delete",
			policy: appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
			},
			expectPodOwnerRef: true,
			expectSetOwnerRef: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)()
			closeFn, rm, informers, c := scSetup(t)
			defer closeFn()
			ns := framework.CreateNamespaceOrDie(c, "test-autodelete-ownerrefs", t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)
			cancel := runControllerAndInformers(rm, informers)
			defer cancel()

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
