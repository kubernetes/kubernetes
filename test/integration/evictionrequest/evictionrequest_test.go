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

package evictionrequest

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	lifecycleapplyv1alpha1 "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	"k8s.io/client-go/informers"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/evictionrequest"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

const (
	pollInterval              = 100 * time.Millisecond
	controllerResponseTimeout = 5 * time.Second

	responderA    = "a.example.com/foo"
	responderB    = "b.example.com/bar"
	fieldManagerA = "a.example.com"
	fieldManagerB = "b.example.com"
	fieldManagerC = "c.example.com"
)

func TestEvictionRequestValidation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.EvictionRequestAPI: true,
		features.GenericWorkload:    true,
	})
	tCtx := evictionRequestControllerSetup(t)

	testCases := []struct {
		name               string
		createPod          bool
		pod                *v1.Pod
		expectedConditions []metav1.Condition
	}{
		{
			name:               "pod not found",
			pod:                mkValidPod("", addPodUID("55fcee67-c0cb-409d-b610-c16d9d765b7b")),
			expectedConditions: failedConditions(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, "not found"),
		},
		{
			name:               "pod UID mismatch",
			pod:                mkValidPod("", addPodUID("55fcee67-c0cb-409d-b610-c16d9d765b7b")),
			expectedConditions: failedConditions(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, "UID mismatch"),
			createPod:          true,
		},
		{
			name:               "pod with schedulingGroup",
			pod:                mkValidPod("", addSchedulingGroup("group")),
			expectedConditions: failedConditions(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, "references a SchedulingGroup"),
			createPod:          true,
		},
	}
	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-responder-skip-processing-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)
			var err error

			pod := tc.pod
			pod.Namespace = ns.Name
			if tc.createPod {
				var tmpUID apimachinerytypes.UID
				if len(pod.UID) > 0 {
					tmpUID = pod.UID
					pod.UID = ""
				}
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
				tCtx.ExpectNoError(err, "create pod")
				// Don't create the request immediately to prevent a race - so let's just observe it first
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get pod")
				pod, err = markPodRunning(tCtx, time.Now(), pod)
				tCtx.ExpectNoError(err, "mark pod running")
				if len(tmpUID) > 0 {
					pod.UID = tmpUID
				}
			}

			evictionRequest := mkValidEvictionRequest(pod)
			evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Create(tCtx, evictionRequest, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction request")

			expectConditions := matchExpectedConditions(tc.expectedConditions...)
			expectGeneration := gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(1))))

			tCtx.Eventually(getEvictionRequest(evictionRequest)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(gomega.And(expectConditions, expectGeneration))
			evictionRequest, err = getEvictionRequest(evictionRequest)(tCtx)
			tCtx.ExpectNoError(err, "get eviction request")
			evictionRequest.Spec.Intent = lifecyclev1alpha1.EvictionRequestIntentWithdrawn
			// trigger update
			// observedGeneration should be managed even after final conditions are set
			expectGeneration = gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(2))))
			evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Update(tCtx, evictionRequest, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "update eviction request")

			tCtx.Eventually(getEvictionRequest(evictionRequest)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(gomega.And(expectConditions, expectGeneration))

			// status should not change after
			tCtx.Consistently(getEvictionRequest(evictionRequest)).WithTimeout(300 * time.Millisecond).WithPolling(pollInterval).Should(gomega.And(expectConditions, expectGeneration))
		})
	}
}

func TestEvictionValidation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.EvictionRequestAPI: true,
		features.GenericWorkload:    true,
	})
	tCtx := evictionRequestControllerSetup(t)

	testCases := []struct {
		name               string
		createPod          bool
		pod                *v1.Pod
		expectedConditions []metav1.Condition
	}{
		{
			name:               "pod with schedulingGroup",
			pod:                mkValidPod("", addSchedulingGroup("group")),
			expectedConditions: failedConditions(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, "references a SchedulingGroup"),
			createPod:          true,
		},
	}
	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-responder-skip-processing-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)
			var err error

			pod := tc.pod
			pod.Namespace = ns.Name
			if tc.createPod {
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
				tCtx.ExpectNoError(err, "create pod")
				// Don't create the request immediately to prevent a race - so let's just observe it first
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get pod")
				pod, err = markPodRunning(tCtx, time.Now(), pod)
				tCtx.ExpectNoError(err, "mark pod running")
			}

			eviction := mkValidEviction(pod)
			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Create(tCtx, eviction, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction")

			expectConditions := matchExpectedConditions(tc.expectedConditions...)
			expectGeneration := gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(1))))

			tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(gomega.And(expectConditions, expectGeneration))
			// status should not change after
			tCtx.Consistently(getEviction(eviction)).WithTimeout(300 * time.Millisecond).WithPolling(pollInterval).Should(gomega.And(expectConditions, expectGeneration))
		})
	}
}

func TestEvictionRequestEvictionProcess(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.EvictionRequestAPI: true,
		features.GenericWorkload:    true,
	})
	tCtx := evictionRequestControllerSetup(t)

	testCases := []struct {
		name                           string
		deletePod                      bool
		terminatePod                   bool
		cancelEviction                 bool
		completeWithoutTermination     bool
		responderLastUpdateMissing     bool
		createNewRequester             bool
		expectedConditions             []metav1.Condition
		expectedFailedCondition        metav1.Condition
		expectedTargetEvictedCondition metav1.Condition
	}{
		{
			name:               "should evict - pod termination",
			terminatePod:       true,
			expectedConditions: evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "terminal"),
		},
		{
			name:               "should evict - pod deletion",
			deletePod:          true,
			expectedConditions: evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "deleted"),
		},
		{
			name:               "should cancel eviction, and allow for triggering it again via the same eviction request",
			cancelEviction:     true,
			deletePod:          true,
			expectedConditions: evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "deleted"),
		},
		{
			name:               "should cancel eviction, and allow for triggering it again via different eviction request",
			cancelEviction:     true,
			deletePod:          true,
			createNewRequester: true,
			expectedConditions: evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "deleted"),
		},
		{
			name:                       "complete responders without pod termination fails eviction, but recovers with a new eviction",
			terminatePod:               true,
			completeWithoutTermination: true,
			expectedConditions:         evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "terminal"),
		},
		{
			name:                       "should evict with pod termination, but last responder update fails",
			terminatePod:               true,
			responderLastUpdateMissing: true,
			expectedConditions:         evictedConditions(lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "terminal"),
		},
	}
	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-responder-skip-processing-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)
			var err error

			// Create Pod
			pod := mkValidPod("", addFinalizer("prevent/deletion"),
				addResponder(responderB, 1000),
				addResponder(responderA, 1001))
			pod.Namespace = ns.Name
			pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create pod")
			// Don't create the request immediately to prevent a race - so let's just observe it first
			pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod")
			pod, err = markPodRunning(tCtx, time.Now(), pod)
			tCtx.ExpectNoError(err, "mark pod running")

			// Create EvictionRequest
			evictionRequest := mkValidEvictionRequest(pod)
			evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Create(tCtx, evictionRequest, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction request")

			// Check EvictionRequest
			expectAwaitingEvictionConditions := matchExpectedConditions(metav1.Condition{
				Type:   string(lifecyclev1alpha1.EvictionConditionTargetEvicted),
				Status: metav1.ConditionFalse,
				Reason: string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
			}, metav1.Condition{

				Type:   string(lifecyclev1alpha1.EvictionConditionFailed),
				Status: metav1.ConditionFalse,
				Reason: string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
			})
			expectEvictionRequestGeneration := gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(1))))

			tCtx.Eventually(getEvictionRequest(evictionRequest)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
				gomega.And(expectAwaitingEvictionConditions, expectEvictionRequestGeneration))

			// Test Eviction creation
			eviction := &lifecyclev1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-1-" + pod.Name,
					Namespace: pod.Namespace,
				},
			}
			expectEvictionSpec := gomega.HaveField("Spec", gomega.Equal(lifecyclev1alpha1.EvictionSpec{
				Target: lifecyclev1alpha1.EvictionTarget{
					Pod: &lifecyclev1alpha1.EvictionPodReference{
						Name: pod.Name,
						UID:  pod.UID,
					},
				},
			}))
			tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(expectEvictionSpec)
			evictions, err := tCtx.Client().LifecycleV1alpha1().Evictions(evictionRequest.Namespace).List(tCtx, metav1.ListOptions{})
			tCtx.ExpectNoError(err, "list evictions")
			tCtx.Expect(evictions.Items).To(gomega.HaveLen(1))
			eviction = new(evictions.Items[0])

			// Test Eviction progress
			expectRequesters := gomega.HaveField("Status.Requesters", gomega.Equal([]lifecyclev1alpha1.Requester{
				{Name: evictionRequest.Spec.Requester, Intent: lifecyclev1alpha1.RequesterIntentEviction},
			}))
			expectEvictionGeneration := gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(1))))
			expectEvictionLabels := gomega.HaveField("Labels", gomega.Equal(map[string]string{
				"foo.example.com/1": "requester",
				responderA:          "responder",
				responderB:          "responder",
				lifecyclev1alpha1.EvictionResponderImperativeEviction: "responder",
			}))
			expectEvictionOwnerRef := gomega.HaveField("OwnerReferences", gomega.Equal([]metav1.OwnerReference{{
				APIVersion: "lifecycle.k8s.io/v1alpha1",
				Kind:       "EvictionRequest",
				Name:       evictionRequest.Name,
				UID:        evictionRequest.UID,
			}}))

			// First Responder selected
			expectTargetRespondersMatcher := matchResponderState(
				lifecyclev1alpha1.ResponderStateActive,
				lifecyclev1alpha1.ResponderStateInactive,
				lifecyclev1alpha1.ResponderStateInactive)
			expectStatusResponders := matchExpectedStatusResponders([]string{responderA, responderB, lifecyclev1alpha1.EvictionResponderImperativeEviction}, responderA)

			tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(gomega.And(
				expectEvictionLabels,
				expectEvictionOwnerRef,
				expectEvictionGeneration,
				expectRequesters,
				expectTargetRespondersMatcher,
				expectStatusResponders,
				expectAwaitingEvictionConditions,
			))
			// First responder progress and completion
			eviction, err = markResponderUpdate(tCtx, fieldManagerA, eviction, responderA, false)
			tCtx.ExpectNoError(err, "eviction responder update")

			eviction, err = markResponderUpdate(tCtx, fieldManagerA, eviction, responderA, true)
			tCtx.ExpectNoError(err, "eviction responder update completion")

			// Second Responder selected
			expectTargetRespondersMatcher = matchResponderState(
				lifecyclev1alpha1.ResponderStateCompleted,
				lifecyclev1alpha1.ResponderStateActive,
				lifecyclev1alpha1.ResponderStateInactive)
			expectStatusResponders = matchExpectedStatusResponders([]string{responderA, responderB, lifecyclev1alpha1.EvictionResponderImperativeEviction}, responderB)

			tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(gomega.And(
				expectEvictionGeneration,
				expectRequesters,
				expectTargetRespondersMatcher,
				expectStatusResponders,
				expectAwaitingEvictionConditions,
			))

			// Second responder progress
			eviction, err = markResponderUpdate(tCtx, fieldManagerB, eviction, responderB, false)
			tCtx.ExpectNoError(err, "eviction responder update")

			switch {
			case tc.cancelEviction:
				// Cancel
				evictionRequest, err = getEvictionRequest(evictionRequest)(tCtx)
				tCtx.ExpectNoError(err, "get eviction request")
				evictionRequest.Spec.Intent = lifecyclev1alpha1.EvictionRequestIntentWithdrawn
				evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Update(tCtx, evictionRequest, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "update eviction request")
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					matchResponderState(
						lifecyclev1alpha1.ResponderStateCompleted,
						lifecyclev1alpha1.ResponderStateCanceled,
						lifecyclev1alpha1.ResponderStateInactive))
				// Observe failed conditions
				expectConditions := matchExpectedConditions(
					failedConditions(lifecyclev1alpha1.EvictionConditionReasonCanceledDueToNoRequesters, "No active requesters")...)
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					gomega.And(expectConditions, expectEvictionGeneration))
				// Same conditions are written to the EvictionRequest.
				expectEvictionRequestGeneration = gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(2))))
				tCtx.Eventually(getEvictionRequest(evictionRequest)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					gomega.And(expectConditions, expectEvictionRequestGeneration))
				// Start eviction again
				if tc.createNewRequester {
					oldEvictionRequest := evictionRequest
					evictionRequest = mkValidEvictionRequest(pod, setERRequester("foo.example.com/2"))
					evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Create(tCtx, evictionRequest, metav1.CreateOptions{})
					tCtx.ExpectNoError(err, "create eviction request")
					expectEvictionRequestGeneration = gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(1))))
					expectEvictionLabels = gomega.HaveField("Labels", gomega.Equal(map[string]string{
						"foo.example.com/1": "requester",
						"foo.example.com/2": "requester",
						responderA:          "responder",
						responderB:          "responder",
						lifecyclev1alpha1.EvictionResponderImperativeEviction: "responder",
					}))
					expectEvictionOwnerRef = gomega.HaveField("OwnerReferences", gomega.And(gomega.ContainElement(gomega.Equal(metav1.OwnerReference{
						APIVersion: "lifecycle.k8s.io/v1alpha1",
						Kind:       "EvictionRequest",
						Name:       oldEvictionRequest.Name,
						UID:        oldEvictionRequest.UID,
					})), gomega.ContainElement(gomega.Equal(metav1.OwnerReference{
						APIVersion: "lifecycle.k8s.io/v1alpha1",
						Kind:       "EvictionRequest",
						Name:       evictionRequest.Name,
						UID:        evictionRequest.UID,
					}))),
					)
				} else {
					evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Get(tCtx, evictionRequest.Name, metav1.GetOptions{})
					tCtx.ExpectNoError(err, "get eviction request")
					evictionRequest.Spec.Intent = lifecyclev1alpha1.EvictionRequestIntentEviction
					evictionRequest, err = tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Update(tCtx, evictionRequest, metav1.UpdateOptions{})
					tCtx.ExpectNoError(err, "update eviction request")
					expectEvictionRequestGeneration = gomega.HaveField("Status.ObservedGeneration", gomega.Equal(new(int64(3))))
				}
			case tc.completeWithoutTermination:
				// Complete 2nd responder
				eviction, err = markResponderUpdate(tCtx, fieldManagerB, eviction, responderB, true)
				tCtx.ExpectNoError(err, "eviction responder update")
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					matchResponderState(
						lifecyclev1alpha1.ResponderStateCompleted,
						lifecyclev1alpha1.ResponderStateCompleted,
						lifecyclev1alpha1.ResponderStateActive))
				// Complete 3rd responder
				eviction, err = markResponderUpdate(tCtx, fieldManagerC, eviction, lifecyclev1alpha1.EvictionResponderImperativeEviction, true)
				tCtx.ExpectNoError(err, "eviction responder update")
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					matchResponderState(
						lifecyclev1alpha1.ResponderStateCompleted,
						lifecyclev1alpha1.ResponderStateCompleted,
						lifecyclev1alpha1.ResponderStateCompleted))
				// Observe failed conditions
				expectConditions := matchExpectedConditions(
					failedConditions(lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "completed without evicting")...)
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					gomega.And(expectConditions, expectEvictionGeneration))
				// EvictionRequest recovers and immediately becomes AwaitingEviction
			}

			// recover from a failed eviction
			if tc.cancelEviction || tc.completeWithoutTermination {
				// New Eviction should be immediately created by the controller
				eviction = &lifecyclev1alpha1.Eviction{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod-2-" + pod.Name,
						Namespace: pod.Namespace,
					},
				}
				// Ensure eviction active
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(expectEvictionSpec)
				eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(evictionRequest.Namespace).Get(tCtx, eviction.Name, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get second eviction")
				tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
					gomega.And(expectEvictionLabels, expectEvictionOwnerRef, matchResponderState(
						lifecyclev1alpha1.ResponderStateActive,
						lifecyclev1alpha1.ResponderStateInactive,
						lifecyclev1alpha1.ResponderStateInactive)))
				// Complete first responder
				eviction, err = markResponderUpdate(tCtx, fieldManagerA, eviction, responderA, true)
				tCtx.ExpectNoError(err, "eviction responder update completion")
			}

			tCtx.Eventually(getEviction(eviction)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
				matchResponderState(
					lifecyclev1alpha1.ResponderStateCompleted,
					lifecyclev1alpha1.ResponderStateActive,
					lifecyclev1alpha1.ResponderStateInactive))

			startDeletion := time.Now()
			if tc.deletePod {
				err = deletePod(tCtx, pod)
				tCtx.ExpectNoError(err, "delete pod")
			} else if tc.terminatePod {
				pod, err = markPodSucceeded(tCtx, pod)
				tCtx.ExpectNoError(err, "mark pod succeeded")
			} else {
				tCtx.Errorf("should complete by either pod termination or deletion")
			}

			// It is possible for responder B to do a delayed completion update in order to observe pod deletion
			// simulate pod observation by the responder
			_, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			if tc.deletePod {
				if err == nil || !apierrors.IsNotFound(err) {
					tCtx.Errorf("expected not found err: %v", err)
				}
			} else {
				tCtx.ExpectNoError(err, "get pod")
			}

			// The following tests can fail if the termination takes too long.
			// See GracefulCompletionDelay in evictionrequest-controller (currently 5s).
			t.Logf("Termination of pod %v took %v", klog.KObj(pod), time.Since(startDeletion))

			if !tc.responderLastUpdateMissing {
				// Second responder complete
				eviction, err = markResponderUpdate(tCtx, fieldManagerB, eviction, responderB, true)
				tCtx.ExpectNoError(err, "eviction responder update")
				expectTargetRespondersMatcher = matchResponderState(
					lifecyclev1alpha1.ResponderStateCompleted,
					lifecyclev1alpha1.ResponderStateCompleted,
					lifecyclev1alpha1.ResponderStateInactive)
			} else {
				expectTargetRespondersMatcher = matchResponderState(
					lifecyclev1alpha1.ResponderStateCompleted,
					lifecyclev1alpha1.ResponderStateInterrupted,
					lifecyclev1alpha1.ResponderStateInactive)
			}

			// Observe completion - controller finalizes status
			expectConditions := matchExpectedConditions(tc.expectedConditions...)

			// 5s is the controller backoff if tc.responderLastUpdateMissing
			tCtx.Eventually(getEviction(eviction)).WithTimeout(10 * time.Second).WithPolling(pollInterval).Should(
				gomega.And(
					expectEvictionLabels,
					expectEvictionOwnerRef,
					expectConditions,
					expectTargetRespondersMatcher,
					expectEvictionGeneration,
				))
			// Same conditions are written to the EvictionRequest.
			tCtx.Eventually(getEvictionRequest(evictionRequest)).WithTimeout(controllerResponseTimeout).WithPolling(pollInterval).Should(
				gomega.And(expectConditions, expectEvictionRequestGeneration))
		})
	}
}

func failedConditions(reason lifecyclev1alpha1.EvictionConditionReason, message string) []metav1.Condition {
	return []metav1.Condition{
		{
			Type:    string(lifecyclev1alpha1.EvictionConditionFailed),
			Status:  metav1.ConditionTrue,
			Reason:  string(reason),
			Message: message,
		},
		{
			Type:   string(lifecyclev1alpha1.EvictionConditionTargetEvicted),
			Status: metav1.ConditionFalse,
			Reason: string(lifecyclev1alpha1.EvictionConditionReasonEvictionFailed),
		},
	}
}
func evictedConditions(reason lifecyclev1alpha1.EvictionConditionReason, message string) []metav1.Condition {
	return []metav1.Condition{
		{
			Type:   string(lifecyclev1alpha1.EvictionConditionFailed),
			Status: metav1.ConditionFalse,
			Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded),
		},
		{
			Type:    string(lifecyclev1alpha1.EvictionConditionTargetEvicted),
			Status:  metav1.ConditionTrue,
			Reason:  string(reason),
			Message: message,
		},
	}
}

func matchExpectedConditions(conditions ...metav1.Condition,
) types.GomegaMatcher {
	var containElems []types.GomegaMatcher
	for _, condition := range conditions {
		containElems = append(containElems, gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(condition.Type),
				"Status":  gomega.Equal(condition.Status),
				"Reason":  gomega.Equal(condition.Reason),
				"Message": gomega.ContainSubstring(condition.Message),
			}),
		))
	}
	return gomega.HaveField("Status.Conditions", gomega.And(containElems...))
}

func matchResponderState(a, b, c lifecyclev1alpha1.ResponderStateType) gomega.OmegaMatcher {
	return gomega.HaveField("Status.TargetResponders", gomega.Equal([]lifecyclev1alpha1.TargetResponder{
		{Name: responderA, Priority: new(int32(1001)), State: a},
		{Name: responderB, Priority: new(int32(1000)), State: b},
		{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: c},
	}))
}

func matchExpectedStatusResponders(targetResponders []string, startedResponder string) types.GomegaMatcher {
	var matchers []types.GomegaMatcher
	for _, responder := range targetResponders {
		fields := gstruct.Fields{
			"Name": gomega.Equal(responder),
		}
		if responder == startedResponder {
			fields["StartTime"] = gomega.Not(gomega.BeNil())
		}
		matchers = append(matchers, gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, fields),
		))
	}
	return gomega.HaveField("Status.Responders", gomega.And(matchers...))
}

func markPodRunning(tCtx ktesting.TContext, time time.Time, pod *v1.Pod) (*v1.Pod, error) {
	pod.Status.Phase = v1.PodRunning
	pod.Status.Conditions = []v1.PodCondition{{
		Type:               v1.PodReady,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Time{Time: time},
	}}
	return tCtx.Client().CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
}

func markPodSucceeded(tCtx ktesting.TContext, pod *v1.Pod) (*v1.Pod, error) {
	pod.Status.Phase = v1.PodSucceeded
	pod.Status.ContainerStatuses = []v1.ContainerStatus{
		{
			State: v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.Now()},
			},
		},
	}
	return tCtx.Client().CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
}

func deletePod(tCtx ktesting.TContext, pod *v1.Pod) error {
	err := tCtx.Client().CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
	if err != nil {
		return err
	}
	// Postpone deletion for the duration of these two calls
	pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod.Finalizers = nil
	_, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Update(tCtx, pod, metav1.UpdateOptions{})
	return err
}

func markResponderUpdate(tCtx ktesting.TContext, fieldManager string, eviction *lifecyclev1alpha1.Eviction, responderName string, isCompleted bool) (*lifecyclev1alpha1.Eviction, error) {
	newResponderStatus := lifecycleapplyv1alpha1.ResponderStatus().
		WithName(responderName).
		WithHeartbeatTime(metav1.Now()).
		WithExpectedCompletionTime(metav1.Time{Time: time.Now().Add(time.Hour)}).
		WithMessage(responderName)
	if isCompleted {
		newResponderStatus.WithCompletionTime(metav1.Now())
	}

	statusApplyUpdate := lifecycleapplyv1alpha1.Eviction(eviction.Name, eviction.Namespace).
		WithStatus(lifecycleapplyv1alpha1.EvictionStatus().WithResponders(newResponderStatus))
	return tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).ApplyStatus(tCtx, statusApplyUpdate, metav1.ApplyOptions{FieldManager: fieldManager})
}

func getEviction(eviction *lifecyclev1alpha1.Eviction) func(tCtx ktesting.TContext) (*lifecyclev1alpha1.Eviction, error) {
	return func(tCtx ktesting.TContext) (*lifecyclev1alpha1.Eviction, error) {
		return tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Get(tCtx, eviction.Name, metav1.GetOptions{})
	}
}

func getEvictionRequest(evictionRequest *lifecyclev1alpha1.EvictionRequest) func(tCtx ktesting.TContext) (*lifecyclev1alpha1.EvictionRequest, error) {
	return func(tCtx ktesting.TContext) (*lifecyclev1alpha1.EvictionRequest, error) {
		return tCtx.Client().LifecycleV1alpha1().EvictionRequests(evictionRequest.Namespace).Get(tCtx, evictionRequest.Name, metav1.GetOptions{})
	}
}

func mkValidEvictionRequest(pod *v1.Pod, tweaks ...func(obj *lifecyclev1alpha1.EvictionRequest)) *lifecyclev1alpha1.EvictionRequest {
	obj := &lifecyclev1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "foo-",
			Namespace:    pod.Namespace,
		},
		Spec: lifecyclev1alpha1.EvictionRequestSpec{
			Target: lifecyclev1alpha1.EvictionRequestTarget{
				Pod: &lifecyclev1alpha1.EvictionRequestPodReference{
					UID:  pod.UID,
					Name: pod.Name,
				},
			},
			Requester: "foo.example.com/1",
			Intent:    lifecyclev1alpha1.EvictionRequestIntentEviction,
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}
	return obj
}

func setERRequester(requester string) func(obj *lifecyclev1alpha1.EvictionRequest) {
	return func(obj *lifecyclev1alpha1.EvictionRequest) {
		obj.Spec.Requester = requester
	}
}

func mkValidEviction(pod *v1.Pod, tweaks ...func(obj *lifecyclev1alpha1.Eviction)) *lifecyclev1alpha1.Eviction {
	obj := &lifecyclev1alpha1.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "evict-",
			Namespace:    pod.Namespace,
		},
		Spec: lifecyclev1alpha1.EvictionSpec{
			Target: lifecyclev1alpha1.EvictionTarget{
				Pod: &lifecyclev1alpha1.EvictionPodReference{
					UID:  pod.UID,
					Name: pod.Name,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}
	return obj
}

func mkValidPod(namespace string, tweaks ...func(obj *v1.Pod)) *v1.Pod {
	obj := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "test-container",
				Image: "busybox",
			}},
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}
	return obj
}

func addPodUID(uid string) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.UID = apimachinerytypes.UID(uid)
	}
}

func addSchedulingGroup(groupName string) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new(groupName)}
	}
}

func addResponder(name string, priority int32) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.Spec.EvictionResponders = append(obj.Spec.EvictionResponders, v1.EvictionResponder{Name: name, Priority: &priority})
	}
}

func addFinalizer(name string) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.Finalizers = append(obj.Finalizers, name)
	}
}

// evictionRequestControllerSetup sets up necessities for evictionrequest-controller integration test, including control plane, apiserver, informers, and clientset
func evictionRequestControllerSetup(t *testing.T) ktesting.TContext {
	tCtx := ktesting.Init(t)
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	flags := framework.DefaultTestServerFlags()
	flags = append(flags, "--runtime-config=lifecycle.k8s.io/v1alpha1=true")
	if utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload) {
		flags = append(flags, "--runtime-config=scheduling.k8s.io/v1beta1=true")
	}

	server, err := kubeapiservertesting.StartTestServer(t, nil, flags, framework.SharedEtcd())
	tCtx.ExpectNoError(err, "start apiserver")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Stopping the apiserver...")
		server.TearDownFn()
	})
	config := restclient.CopyConfig(server.ClientConfig)
	tCtx = tCtx.WithRESTConfig(config)

	informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Cancel("stopping informers")
		informerFactory.Shutdown()
	})

	responderController, err := evictionrequest.NewController(
		tCtx,
		names.EvictionRequestController,
		informerFactory.Lifecycle().V1alpha1().Evictions(),
		informerFactory.Lifecycle().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		tCtx.Client(),
	)
	tCtx.ExpectNoError(err, "create evictionrequest-controller")

	informerFactory.StartWithContext(tCtx)
	var wg sync.WaitGroup

	wg.Go(func() {
		responderController.Run(tCtx, 1) /* one worker to get more readable log output without interleaving */
	})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Cancel("test is done")
		wg.Wait()
	})

	// since this method starts the controller in a separate goroutine
	// and the tests don't check /readyz there is no way
	// the tests can tell it is safe to call the server and requests won't be rejected
	// thus we wait until caches have synced
	informerFactory.WaitForCacheSyncWithContext(tCtx)

	return tCtx
}
