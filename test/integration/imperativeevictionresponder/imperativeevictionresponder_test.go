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

package imperativeevictionresponder

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/imperativeevictionresponder"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
)

func TestResponderSkipProcessing(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)
	tCtx := imperativeEvictionResponderSetup(t)

	tests := []struct {
		name                  string
		responderLabelValue   lifecyclev1alpha1.EvictionParticipantRole
		prepareEvictionStatus func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus
		expectCompletionTime  bool
	}{
		{
			name:                "should skip processing for a non-annotated eviction",
			responderLabelValue: "",
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateActive)
			},
		},
		{
			name:                "should skip processing for incorrectly annotated eviction",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleRequester,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateActive)
			},
		},
		{
			name:                "should skip processing for non observed eviction",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleResponder,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateActive, setObservedGeneration(nil))
			},
		},
		{
			name:                "should skip processing for failed eviction",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleResponder,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateInactive, addConditions(metav1.Condition{
					Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionTrue, ObservedGeneration: 1, LastTransitionTime: metav1.Now(), Reason: string(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid), Message: "message",
				}))
			},
		},
		{
			name:                "should skip processing for evicted eviction",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleResponder,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateInactive, addConditions(metav1.Condition{
					Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionTrue, ObservedGeneration: 1, LastTransitionTime: metav1.Now(), Reason: string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted), Message: "message",
				}))
			},
		},
		{
			name:                "should skip processing for inactive eviction",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleResponder,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return lifecyclev1alpha1.EvictionStatus{
					ObservedGeneration: new(int64(1)),
					TargetResponders: []lifecyclev1alpha1.TargetResponder{
						{Name: "foo/bar", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateActive},
						{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
					},
					Responders: []lifecyclev1alpha1.ResponderStatus{
						{Name: "foo/bar", StartTime: new(metav1.Time{Time: passiveClock.Now()})},
						{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction},
					},
				}
			},
		},
		{
			name:                "should skip processing for eviction with completionTime",
			responderLabelValue: lifecyclev1alpha1.EvictionParticipantRoleResponder,
			prepareEvictionStatus: func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus {
				return lifecyclev1alpha1.EvictionStatus{
					ObservedGeneration: new(int64(1)),
					TargetResponders: []lifecyclev1alpha1.TargetResponder{
						{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateActive},
					},
					Responders: []lifecyclev1alpha1.ResponderStatus{
						{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Time{Time: passiveClock.Now()}), CompletionTime: new(metav1.Time{Time: passiveClock.Now().Add(time.Second)})},
					},
				}
			},
			expectCompletionTime: true,
		},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-responder-skip-processing-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)

			passiveClock := testing2.NewFakeClock(time.Now())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: ns.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "test-container",
						Image: "busybox",
					}},
				},
			}
			pod, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create pod")
			eviction := mkValidEviction(pod)
			if len(tc.responderLabelValue) > 0 {
				eviction.ObjectMeta.Labels = map[string]string{
					"app": "bar",
					lifecyclev1alpha1.EvictionResponderImperativeEviction: string(tc.responderLabelValue),
				}
			}

			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Create(tCtx, eviction, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction")

			// simulate the evictionrequest-controller
			newStatus := tc.prepareEvictionStatus(passiveClock)
			eviction.Status = newStatus
			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).UpdateStatus(tCtx, eviction, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "update eviction:")

			gomegaCompletionTime := gomega.BeZero()
			if tc.expectCompletionTime {
				gomegaCompletionTime = gomega.Not(gomega.BeZero())
			}

			tCtx.Consistently(getEviction(eviction)).WithTimeout(time.Second * 3).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
					"HeartbeatTime":          gomega.BeZero(),
					"ExpectedCompletionTime": gomega.BeZero(),
					"CompletionTime":         gomegaCompletionTime,
					"Message":                gomega.BeNil(),
				}),
			)))
		})
	}
}

func TestResponderEviction(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)
	tCtx := imperativeEvictionResponderSetup(t)

	tests := []struct {
		name           string
		testPDB        bool
		cancelEviction bool
	}{
		{
			name: "normal eviction",
		},
		{
			name:    "pdb stalls eviction",
			testPDB: true,
		},
		{
			name:           "pdb stalls eviction and cancels eviction",
			testPDB:        true,
			cancelEviction: true,
		},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-responder-eviction-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)

			passiveClock := testing2.NewFakeClock(time.Now())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: ns.Name,
					Labels: map[string]string{
						"app": "test-responder",
					},
					Finalizers: []string{
						"eviction-responder/test",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "test-container",
						Image: "busybox",
					}},
				},
			}
			var err error
			pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create pod")

			pod.Status.Phase = v1.PodRunning
			pod.Status.Conditions = []v1.PodCondition{{
				Type:               v1.PodReady,
				Status:             v1.ConditionTrue,
				LastTransitionTime: metav1.Time{Time: passiveClock.Now()},
			}}
			pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "mark pod running")

			pdb := &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pdb",
					Namespace: ns.Name,
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &intstr.IntOrString{
						Type:   intstr.Int,
						IntVal: 1,
					},
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "test-responder"},
					},
				},
			}
			if tc.testPDB {
				pdb, err = tCtx.Client().PolicyV1().PodDisruptionBudgets(ns.Name).Create(tCtx, pdb, metav1.CreateOptions{})
				tCtx.ExpectNoError(err, "create pdb")
				pdb.Status.ObservedGeneration = 1
				pdb.Status.CurrentHealthy = 1
				pdb.Status.DesiredHealthy = 1
				pdb, err = tCtx.Client().PolicyV1().PodDisruptionBudgets(pdb.Namespace).UpdateStatus(tCtx, pdb, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "update pdb")
			}

			eviction := mkValidEviction(pod, setLabels(map[string]string{
				lifecyclev1alpha1.EvictionResponderImperativeEviction: string(lifecyclev1alpha1.EvictionParticipantRoleResponder),
			}))
			status := *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateActive)

			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Create(tCtx, eviction, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction")

			// simulate the evictionrequest-controller
			eviction.Status = status
			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).UpdateStatus(tCtx, eviction, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "update eviction")

			exponentialBackoffTimeout := 3 * time.Second // constant
			if tc.testPDB {
				tCtx.Eventually(getEviction(eviction)).WithTimeout(exponentialBackoffTimeout).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
					gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
						"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
						"StartTime":              gomega.Not(gomega.BeZero()),
						"ExpectedCompletionTime": gomega.BeZero(),
						"CompletionTime":         gomega.BeZero(),
						"HeartbeatTime":          gomega.Not(gomega.BeZero()),
						"Message":                gomega.HaveValue(gomega.ContainSubstring("pod deletion via the /eviction subresource failed (attempts=1)")),
					}),
				)))
				exponentialBackoffTimeout += 5 * time.Second // f(0) + constant

				if tc.cancelEviction {
					eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Get(tCtx, eviction.Name, metav1.GetOptions{})
					tCtx.ExpectNoError(err, "get eviction")
					eviction.Status.TargetResponders[0].State = lifecyclev1alpha1.ResponderStateCanceled
					eviction.Status.Conditions = []metav1.Condition{
						{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionTrue, ObservedGeneration: 1, LastTransitionTime: metav1.Time{Time: passiveClock.Now()}, Reason: string(lifecyclev1alpha1.EvictionConditionReasonCanceledDueToNoRequesters), Message: "message"},
					}
					eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).UpdateStatus(tCtx, eviction, metav1.UpdateOptions{})
					tCtx.ExpectNoError(err, "update eviction")
					err = tCtx.Client().PolicyV1().PodDisruptionBudgets(ns.Name).Delete(tCtx, pdb.Name, metav1.DeleteOptions{})
					tCtx.ExpectNoError(err, "delete pdb")

					tCtx.Consistently(getEviction(eviction)).WithTimeout(exponentialBackoffTimeout).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
						gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
							"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
							"StartTime":              gomega.Not(gomega.BeZero()),
							"ExpectedCompletionTime": gomega.BeZero(),
							"CompletionTime":         gomega.BeZero(),
							"HeartbeatTime":          gomega.Not(gomega.BeZero()),
							"Message":                gomega.HaveValue(gomega.ContainSubstring("pod deletion via the /eviction subresource failed (attempts=1)")),
						}),
					)))
					return
				}
				tCtx.Eventually(getEviction(eviction)).WithTimeout(exponentialBackoffTimeout).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
					gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
						"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
						"StartTime":              gomega.Not(gomega.BeZero()),
						"ExpectedCompletionTime": gomega.BeZero(),
						"CompletionTime":         gomega.BeZero(),
						"HeartbeatTime":          gomega.Not(gomega.BeZero()),
						"Message":                gomega.HaveValue(gomega.ContainSubstring("pod deletion via the /eviction subresource failed (attempts=2)")),
					}),
				)))
				err = tCtx.Client().PolicyV1().PodDisruptionBudgets(ns.Name).Delete(tCtx, pdb.Name, metav1.DeleteOptions{})
				tCtx.ExpectNoError(err, "delete pdb")
				exponentialBackoffTimeout += 5 * time.Second // f(1) + constant
			}
			tCtx.Eventually(getEviction(eviction)).WithTimeout(exponentialBackoffTimeout * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
					"StartTime":              gomega.Not(gomega.BeZero()),
					"HeartbeatTime":          gomega.Not(gomega.BeZero()),
					"ExpectedCompletionTime": gomega.Not(gomega.BeZero()),
					"CompletionTime":         gomega.BeZero(),
					"Message":                gomega.HaveValue(gomega.ContainSubstring("pod has been been marked for deletion via the /eviction subresource and is being terminated gracefully")),
				}),
			)))

			pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod")
			pod.Status.Phase = v1.PodSucceeded
			_, err = tCtx.Client().CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "mark pod running")

			tCtx.Eventually(getEviction(eviction)).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
					"StartTime":              gomega.Not(gomega.BeZero()),
					"HeartbeatTime":          gomega.Not(gomega.BeZero()),
					"ExpectedCompletionTime": gomega.Not(gomega.BeZero()),
					"CompletionTime":         gomega.Not(gomega.BeZero()),
					"Message":                gomega.HaveValue(gomega.ContainSubstring("pod has been deleted and fully terminated (pod phase=\"Succeeded\")")),
				}),
			)))
		})
	}
}

func TestThirdPartyTerminationDetection(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)
	tCtx := imperativeEvictionResponderSetup(t)

	tests := []struct {
		name                         string
		shouldCreatePod              bool
		responderLabelValue          lifecyclev1alpha1.EvictionParticipantRole
		prepareEvictionStatus        func(passiveClock clock.PassiveClock) lifecyclev1alpha1.EvictionStatus
		expectExpectedCompletionTime bool
		expectCompletionTime         bool
		expectedMessage              string
	}{
		{
			name:            "pod deleted",
			shouldCreatePod: false,
			expectedMessage: "pod has been deleted and fully terminated",
		},
		{
			name:            "pod terminated",
			shouldCreatePod: true,
			expectedMessage: "pod has been fully terminated (pod phase=\"Failed\")",
		},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(tCtx.Client(), fmt.Sprintf("test-third-party-termination-detection-%d", i), t)
			defer framework.DeleteNamespaceOrDie(tCtx.Client(), ns, t)

			passiveClock := testing2.NewFakeClock(time.Now())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: ns.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "test-container",
						Image: "busybox",
					}},
				},
			}
			var err error
			if tc.shouldCreatePod {
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
				tCtx.ExpectNoError(err, "create pod")

				pod.Status.Phase = v1.PodFailed
				pod, err = tCtx.Client().CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "mark pod running")
			} else {
				pod.UID = "ef3f28c5-176a-41ec-a0f9-d069cad9e0ff"
			}
			eviction := mkValidEviction(pod, setLabels(map[string]string{
				lifecyclev1alpha1.EvictionResponderImperativeEviction: string(lifecyclev1alpha1.EvictionParticipantRoleResponder),
			}))
			status := *mkValidActiveEvictionStatus(passiveClock, lifecyclev1alpha1.ResponderStateActive)

			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Create(tCtx, eviction, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create eviction")

			// simulate the evictionrequest-controller
			eviction.Status = status
			eviction, err = tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).UpdateStatus(tCtx, eviction, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "update eviction")

			tCtx.Eventually(getEviction(eviction)).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.HaveField("Status.Responders", gomega.ContainElement(
				gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
					"Name":                   gomega.Equal(lifecyclev1alpha1.EvictionResponderImperativeEviction),
					"StartTime":              gomega.Not(gomega.BeZero()),
					"HeartbeatTime":          gomega.Not(gomega.BeZero()),
					"ExpectedCompletionTime": gomega.Not(gomega.BeZero()),
					"CompletionTime":         gomega.Not(gomega.BeZero()),
					"Message":                gomega.HaveValue(gomega.ContainSubstring(tc.expectedMessage)),
				}),
			)))
		})
	}
}

// imperativeEvictionResponderSetup sets up necessities for imperative-eviction-responder-controller integration test, including control plane, apiserver, informers, and clientset
func imperativeEvictionResponderSetup(t *testing.T) ktesting.TContext {
	tCtx := ktesting.Init(t)
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	flags := framework.DefaultTestServerFlags()
	flags = append(flags, "--runtime-config=lifecycle.k8s.io/v1alpha1=true")

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

	responderController, err := imperativeevictionresponder.NewController(
		tCtx,
		names.ImperativeEvictionResponderController,
		informerFactory.Lifecycle().V1alpha1().Evictions(),
		informerFactory.Core().V1().Pods(),
		tCtx.Client(),
	)
	tCtx.ExpectNoError(err, "create responder controller")

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

func getEviction(eviction *lifecyclev1alpha1.Eviction) func(tCtx ktesting.TContext) (*lifecyclev1alpha1.Eviction, error) {
	return func(tCtx ktesting.TContext) (*lifecyclev1alpha1.Eviction, error) {
		return tCtx.Client().LifecycleV1alpha1().Evictions(eviction.Namespace).Get(tCtx, eviction.Name, metav1.GetOptions{})
	}
}

func mkValidEviction(pod *v1.Pod, tweaks ...func(obj *lifecyclev1alpha1.Eviction)) *lifecyclev1alpha1.Eviction {
	obj := &lifecyclev1alpha1.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: pod.Namespace,
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
func setLabels(labels map[string]string) func(obj *lifecyclev1alpha1.Eviction) {
	return func(obj *lifecyclev1alpha1.Eviction) {
		obj.ObjectMeta.Labels = labels

	}
}

func mkValidActiveEvictionStatus(passiveClock clock.PassiveClock, state lifecyclev1alpha1.ResponderStateType, tweaks ...func(obj *lifecyclev1alpha1.EvictionStatus)) *lifecyclev1alpha1.EvictionStatus {
	obj := &lifecyclev1alpha1.EvictionStatus{
		ObservedGeneration: new(int64(1)),
		TargetResponders: []lifecyclev1alpha1.TargetResponder{
			{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: state},
		},
		Responders: []lifecyclev1alpha1.ResponderStatus{
			{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Time{Time: passiveClock.Now()})},
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}

	return obj
}
func setObservedGeneration(observedGeneration *int64) func(obj *lifecyclev1alpha1.EvictionStatus) {
	return func(obj *lifecyclev1alpha1.EvictionStatus) {
		obj.ObservedGeneration = observedGeneration

	}
}
func addConditions(conditions ...metav1.Condition) func(obj *lifecyclev1alpha1.EvictionStatus) {
	return func(obj *lifecyclev1alpha1.EvictionStatus) {
		obj.Conditions = append(obj.Conditions, conditions...)

	}
}
