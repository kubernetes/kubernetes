/*
Copyright 2022 The Kubernetes Authors.

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

package podgc

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/podgc"
	"k8s.io/kubernetes/pkg/features"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// TestPodGcOrphanedPodsWithFinalizer tests deletion of orphaned pods
func TestPodGcOrphanedPodsWithFinalizer(t *testing.T) {
	tests := map[string]struct {
		enablePodDisruptionConditions bool
		enableJobPodReplacementPolicy bool
		phase                         v1.PodPhase
		wantPhase                     v1.PodPhase
		wantDisruptionTarget          *v1.PodCondition
	}{
		"PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodPending,
			wantPhase:                     v1.PodFailed,
			wantDisruptionTarget: &v1.PodCondition{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByPodGC",
				Message: "PodGC: node no longer exists",
			},
		},
		"PodDisruptionConditions and PodReplacementPolicy enabled": {
			enablePodDisruptionConditions: true,
			enableJobPodReplacementPolicy: true,
			phase:                         v1.PodPending,
			wantPhase:                     v1.PodFailed,
			wantDisruptionTarget: &v1.PodCondition{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByPodGC",
				Message: "PodGC: node no longer exists",
			},
		},
		"Only PodReplacementPolicy enabled; no PodDisruptionCondition": {
			enablePodDisruptionConditions: false,
			enableJobPodReplacementPolicy: true,
			phase:                         v1.PodPending,
			wantPhase:                     v1.PodFailed,
		},
		"PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			phase:                         v1.PodPending,
			wantPhase:                     v1.PodPending,
		},
		"PodDisruptionConditions enabled; succeeded pod": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodSucceeded,
			wantPhase:                     v1.PodSucceeded,
		},
		"PodDisruptionConditions enabled; failed pod": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodFailed,
			wantPhase:                     v1.PodFailed,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDisruptionConditions, test.enablePodDisruptionConditions)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, test.enableJobPodReplacementPolicy)
			testCtx := setup(t, "podgc-orphaned")
			cs := testCtx.ClientSet

			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node",
				},
				Spec: v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
						},
					},
				},
			}
			node, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create node '%v', err: %v", node.Name, err)
			}

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "testpod",
					Namespace:  testCtx.NS.Name,
					Finalizers: []string{"test.k8s.io/finalizer"},
				},
				Spec: v1.PodSpec{
					NodeName: node.Name,
					Containers: []v1.Container{
						{Name: "foo", Image: "bar"},
					},
				},
			}

			pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error %v, while creating pod: %v", err, klog.KObj(pod))
			}
			defer testutils.RemovePodFinalizers(testCtx.Ctx, testCtx.ClientSet, t, *pod)

			pod.Status.Phase = test.phase
			if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).UpdateStatus(testCtx.Ctx, pod, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("Error %v, while setting phase %v for pod: %v", err, test.phase, klog.KObj(pod))
			}

			// we delete the node to orphan the pod
			err = cs.CoreV1().Nodes().Delete(testCtx.Ctx, pod.Spec.NodeName, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("Failed to delete node: %v, err: %v", pod.Spec.NodeName, err)
			}
			err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, time.Second*15, true, testutils.PodIsGettingEvicted(cs, pod.Namespace, pod.Name))
			if err != nil {
				t.Fatalf("Error '%v' while waiting for the pod '%v' to be terminating", err, klog.KObj(pod))
			}
			pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Error: '%v' while updating pod info: '%v'", err, klog.KObj(pod))
			}
			_, gotDisruptionTarget := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
			if diff := cmp.Diff(test.wantDisruptionTarget, gotDisruptionTarget, cmpopts.IgnoreFields(v1.PodCondition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Pod %v has unexpected DisruptionTarget condition: %s", klog.KObj(pod), diff)
			}
			if pod.Status.Phase != test.wantPhase {
				t.Errorf("Unexpected phase for pod %q. Got: %q, want: %q", klog.KObj(pod), pod.Status.Phase, test.wantPhase)
			}
		})
	}
}

// TestTerminatingOnOutOfServiceNode tests deletion pods terminating on out-of-service nodes
func TestTerminatingOnOutOfServiceNode(t *testing.T) {
	tests := map[string]struct {
		enablePodDisruptionConditions bool
		enableJobPodReplacementPolicy bool
		withFinalizer                 bool
		wantPhase                     v1.PodPhase
	}{
		"pod has phase changed to Failed when PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			withFinalizer:                 true,
			wantPhase:                     v1.PodFailed,
		},
		"pod has phase unchanged when PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			withFinalizer:                 true,
			wantPhase:                     v1.PodPending,
		},
		"pod is getting deleted when no finalizer and PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			withFinalizer:                 false,
		},
		"pod is getting deleted when no finalizer and PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			withFinalizer:                 false,
		},
		"pod has phase changed when PodDisruptionConditions disabled, but JobPodReplacementPolicy enabled": {
			enablePodDisruptionConditions: false,
			enableJobPodReplacementPolicy: true,
			withFinalizer:                 true,
			wantPhase:                     v1.PodFailed,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDisruptionConditions, test.enablePodDisruptionConditions)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeOutOfServiceVolumeDetach, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, test.enableJobPodReplacementPolicy)
			testCtx := setup(t, "podgc-out-of-service")
			cs := testCtx.ClientSet

			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node",
				},
				Spec: v1.NodeSpec{},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionFalse,
						},
					},
				},
			}
			node, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create node '%v', err: %v", node.Name, err)
			}

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "testpod",
					Namespace: testCtx.NS.Name,
				},
				Spec: v1.PodSpec{
					NodeName: node.Name,
					Containers: []v1.Container{
						{Name: "foo", Image: "bar"},
					},
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
				},
			}
			if test.withFinalizer {
				pod.ObjectMeta.Finalizers = []string{"test.k8s.io/finalizer"}
			}

			pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error %v, while creating pod: %v", err, klog.KObj(pod))
			}
			if test.withFinalizer {
				defer testutils.RemovePodFinalizers(testCtx.Ctx, testCtx.ClientSet, t, *pod)
			}

			// trigger termination of the pod, but with long grace period so that it is not removed immediately
			err = cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](300)})
			if err != nil {
				t.Fatalf("Error: '%v' while deleting pod: '%v'", err, klog.KObj(pod))
			}
			// wait until the pod is terminating
			err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, time.Second*15, true, testutils.PodIsGettingEvicted(cs, pod.Namespace, pod.Name))
			if err != nil {
				t.Fatalf("Error '%v' while waiting for the pod '%v' to be terminating", err, klog.KObj(pod))
			}
			// taint the node with the out-of-service taint
			err = testutils.AddTaintToNode(cs, pod.Spec.NodeName, v1.Taint{Key: v1.TaintNodeOutOfService, Value: "", Effect: v1.TaintEffectNoExecute})
			if err != nil {
				t.Fatalf("Failed to taint node: %v, err: %v", pod.Spec.NodeName, err)
			}
			if test.withFinalizer {
				// wait until the pod phase is set as expected
				err = wait.Poll(time.Second, time.Second*15, func() (bool, error) {
					var e error
					pod, e = cs.CoreV1().Pods(pod.Namespace).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
					if e != nil {
						return true, e
					}
					return test.wantPhase == pod.Status.Phase, nil
				})
				if err != nil {
					t.Errorf("Error %q while waiting for the pod %q to be in expected phase", err, klog.KObj(pod))
				}
				_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
				if cond != nil {
					t.Errorf("Pod %q has an unexpected condition: %q", klog.KObj(pod), v1.DisruptionTarget)
				}
			} else {
				// wait until the pod is deleted
				err = wait.PollImmediate(time.Second, time.Second*15, func() (bool, error) {
					var e error
					pod, e = cs.CoreV1().Pods(pod.Namespace).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
					if e == nil {
						return pod == nil, nil
					}
					// there was an error
					if apierrors.IsNotFound(e) {
						return true, nil
					}
					return false, e
				})
				if err != nil {
					t.Errorf("Error %q while waiting for the pod %q to be deleted", err, klog.KObj(pod))
				}
			}
		})
	}
}

// TestPodGcForPodsWithDuplicatedFieldKeys regression test for https://issues.k8s.io/118261
func TestPodGcForPodsWithDuplicatedFieldKeys(t *testing.T) {
	tests := map[string]struct {
		pod                  *v1.Pod
		wantDisruptionTarget *v1.PodCondition
	}{
		"Orphan pod with duplicated env vars": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "testpod",
					Finalizers: []string{"test.k8s.io/finalizer"},
				},
				Spec: v1.PodSpec{
					NodeName: "non-existing-node",
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "bar",
							Env: []v1.EnvVar{
								{
									Name:  "XYZ",
									Value: "1",
								},
								{
									Name:  "XYZ",
									Value: "2",
								},
							},
						},
					},
				},
			},
			wantDisruptionTarget: &v1.PodCondition{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByPodGC",
				Message: "PodGC: node no longer exists",
			},
		},
		"Orphan pod with duplicated ports; scenario from https://issues.k8s.io/113482": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "testpod",
					Finalizers: []string{"test.k8s.io/finalizer"},
				},
				Spec: v1.PodSpec{
					NodeName: "non-existing-node",
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "bar",
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 93,
									HostPort:      9376,
								},
								{
									ContainerPort: 93,
									HostPort:      9377,
								},
							},
						},
					},
				},
			},
			wantDisruptionTarget: &v1.PodCondition{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByPodGC",
				Message: "PodGC: node no longer exists",
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDisruptionConditions, true)
			testCtx := setup(t, "podgc-orphaned")
			cs := testCtx.ClientSet

			pod := test.pod
			pod.Namespace = testCtx.NS.Namespace
			pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Error %v, while creating pod: %v", err, klog.KObj(pod))
			}
			defer testutils.RemovePodFinalizers(testCtx.Ctx, testCtx.ClientSet, t, *pod)

			// getting evicted due to NodeName being "non-existing-node"
			err = wait.PollUntilContextTimeout(testCtx.Ctx, time.Second, time.Second*15, true, testutils.PodIsGettingEvicted(cs, pod.Namespace, pod.Name))
			if err != nil {
				t.Fatalf("Error '%v' while waiting for the pod '%v' to be terminating", err, klog.KObj(pod))
			}
			pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Error: '%v' while updating pod info: '%v'", err, klog.KObj(pod))
			}
			_, gotDisruptionTarget := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
			if diff := cmp.Diff(test.wantDisruptionTarget, gotDisruptionTarget, cmpopts.IgnoreFields(v1.PodCondition{}, "LastTransitionTime")); diff != "" {
				t.Errorf("Pod %v has unexpected DisruptionTarget condition: %s", klog.KObj(pod), diff)
			}
			if gotDisruptionTarget != nil && gotDisruptionTarget.LastTransitionTime.IsZero() {
				t.Errorf("Pod %v has DisruptionTarget condition without LastTransitionTime", klog.KObj(pod))
			}
			if pod.Status.Phase != v1.PodFailed {
				t.Errorf("Unexpected phase for pod %q. Got: %q, want: %q", klog.KObj(pod), pod.Status.Phase, v1.PodFailed)
			}
		})
	}
}

func setup(t *testing.T, name string) *testutils.TestContext {
	testCtx := testutils.InitTestAPIServer(t, name, nil)
	externalInformers := informers.NewSharedInformerFactory(testCtx.ClientSet, time.Second)

	podgc := podgc.NewPodGCInternal(testCtx.Ctx,
		testCtx.ClientSet,
		externalInformers.Core().V1().Pods(),
		externalInformers.Core().V1().Nodes(),
		0,
		500*time.Millisecond,
		time.Second)

	// Waiting for all controllers to sync
	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

	go podgc.Run(testCtx.Ctx)
	return testCtx
}
