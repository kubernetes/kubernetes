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
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	corev1helpers "k8s.io/component-helpers/node/corev1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
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
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
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
				  "image": "k8s.gcr.io/nginx-slim:0.8",
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
	s, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-spec-replicas-change", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(rm, informers)
	defer close(stopCh)

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
	s, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-deleting-and-failed-pods", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(rm, informers)
	defer close(stopCh)

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
		enabled        bool
	}{
		{
			name:           "When feature gate is enabled, only certain replicas would become active",
			totalReplicas:  4,
			readyReplicas:  3,
			activeReplicas: 2,
			enabled:        true,
		},
		{
			name:           "When feature gate is disabled, all the ready replicas would become active",
			totalReplicas:  4,
			readyReplicas:  3,
			activeReplicas: 3,
			enabled:        false,
		},
	}
	for _, test := range tests {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetMinReadySeconds, test.enabled)()
		s, closeFn, rm, informers, c := scSetup(t)
		defer closeFn()
		ns := framework.CreateTestingNamespace("test-available-pods", s, t)
		defer framework.DeleteTestingNamespace(ns, s, t)
		stopCh := runControllerAndInformers(rm, informers)
		defer close(stopCh)

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
			_, condition := corev1helpers.GetPodCondition(&pod.Status, v1.PodReady)
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
