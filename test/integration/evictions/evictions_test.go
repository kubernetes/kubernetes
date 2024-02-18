/*
Copyright 2015 The Kubernetes Authors.

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

package evictions

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	policyv1client "k8s.io/client-go/kubernetes/typed/policy/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	numOfEvictions = 10
)

// TestConcurrentEvictionRequests is to make sure pod disruption budgets (PDB) controller is able to
// handle concurrent eviction requests. Original issue:#37605
func TestConcurrentEvictionRequests(t *testing.T) {
	podNameFormat := "test-pod-%d"

	tCtx := ktesting.Init(t)
	closeFn, rm, informers, _, clientSet := rmSetup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "concurrent-eviction-requests", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
	defer tCtx.Cancel("test has completed")

	informers.Start(tCtx.Done())
	go rm.Run(tCtx)

	var gracePeriodSeconds int64 = 30
	deleteOption := metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}

	// Generate numOfEvictions pods to evict
	for i := 0; i < numOfEvictions; i++ {
		podName := fmt.Sprintf(podNameFormat, i)
		pod := newPod(podName)

		if _, err := clientSet.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}
		pod.Status.Phase = v1.PodRunning
		addPodConditionReady(pod)
		if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), numOfEvictions, v1.PodRunning)

	pdb := newPDB()
	if _, err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).Create(context.TODO(), pdb, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create PodDisruptionBudget: %v", err)
	}

	waitPDBStable(t, clientSet, ns.Name, pdb.Name, numOfEvictions)

	var numberPodsEvicted uint32
	errCh := make(chan error, 3*numOfEvictions)
	var wg sync.WaitGroup
	// spawn numOfEvictions goroutines to concurrently evict the pods
	for i := 0; i < numOfEvictions; i++ {
		wg.Add(1)
		go func(id int, errCh chan error) {
			defer wg.Done()
			podName := fmt.Sprintf(podNameFormat, id)
			eviction := newV1Eviction(ns.Name, podName, deleteOption)

			err := wait.PollImmediate(5*time.Second, 60*time.Second, func() (bool, error) {
				e := clientSet.PolicyV1().Evictions(ns.Name).Evict(context.TODO(), eviction)
				switch {
				case apierrors.IsTooManyRequests(e):
					return false, nil
				case apierrors.IsConflict(e):
					return false, fmt.Errorf("Unexpected Conflict (409) error caused by failing to handle concurrent PDB updates: %v", e)
				case e == nil:
					return true, nil
				default:
					return false, e
				}
			})

			if err != nil {
				errCh <- err
				// should not return here otherwise we would leak the pod
			}

			_, err = clientSet.CoreV1().Pods(ns.Name).Get(context.TODO(), podName, metav1.GetOptions{})
			switch {
			case apierrors.IsNotFound(err):
				atomic.AddUint32(&numberPodsEvicted, 1)
				// pod was evicted and deleted so return from goroutine immediately
				return
			case err == nil:
				// this shouldn't happen if the pod was evicted successfully
				errCh <- fmt.Errorf("Pod %q is expected to be evicted", podName)
			default:
				errCh <- err
			}

			// delete pod which still exists due to error
			e := clientSet.CoreV1().Pods(ns.Name).Delete(context.TODO(), podName, deleteOption)
			if e != nil {
				errCh <- e
			}

		}(i, errCh)
	}

	wg.Wait()

	close(errCh)
	var errList []error
	if err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).Delete(context.TODO(), pdb.Name, deleteOption); err != nil {
		errList = append(errList, fmt.Errorf("Failed to delete PodDisruptionBudget: %v", err))
	}
	for err := range errCh {
		errList = append(errList, err)
	}
	if len(errList) > 0 {
		t.Fatal(utilerrors.NewAggregate(errList))
	}

	if atomic.LoadUint32(&numberPodsEvicted) != numOfEvictions {
		t.Fatalf("fewer number of successful evictions than expected : %d", numberPodsEvicted)
	}
}

// TestTerminalPodEviction ensures that PDB is not checked for terminal pods.
func TestTerminalPodEviction(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, rm, informers, _, clientSet := rmSetup(tCtx, t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "terminalpod-eviction", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
	defer tCtx.Cancel("test has completed")

	informers.Start(tCtx.Done())
	go rm.Run(tCtx)

	var gracePeriodSeconds int64 = 30
	deleteOption := metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}
	pod := newPod("test-terminal-pod1")
	if _, err := clientSet.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}

	pod.Status.Phase = v1.PodSucceeded
	addPodConditionReady(pod)
	if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), 1, v1.PodSucceeded)

	pdb := newPDB()
	if _, err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).Create(context.TODO(), pdb, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create PodDisruptionBudget: %v", err)
	}

	waitPDBStable(t, clientSet, ns.Name, pdb.Name, 1)

	pdbList, err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Error while listing pod disruption budget")
	}
	oldPdb := pdbList.Items[0]
	eviction := newV1Eviction(ns.Name, pod.Name, deleteOption)
	err = wait.PollImmediate(5*time.Second, 60*time.Second, func() (bool, error) {
		e := clientSet.PolicyV1().Evictions(ns.Name).Evict(context.TODO(), eviction)
		switch {
		case apierrors.IsTooManyRequests(e):
			return false, nil
		case apierrors.IsConflict(e):
			return false, fmt.Errorf("Unexpected Conflict (409) error caused by failing to handle concurrent PDB updates: %v", e)
		case e == nil:
			return true, nil
		default:
			return false, e
		}
	})
	if err != nil {
		t.Fatalf("Eviction of pod failed %v", err)
	}
	pdbList, err = clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Error while listing pod disruption budget")
	}
	newPdb := pdbList.Items[0]
	// We shouldn't see an update in pod disruption budget status' generation number as we are evicting terminal pods without checking for pod disruption.
	if !reflect.DeepEqual(newPdb.Status.ObservedGeneration, oldPdb.Status.ObservedGeneration) {
		t.Fatalf("Expected the pdb generation to be of same value %v but got %v", newPdb.Status.ObservedGeneration, oldPdb.Status.ObservedGeneration)
	}

	if err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).Delete(context.TODO(), pdb.Name, deleteOption); err != nil {
		t.Fatalf("Failed to delete pod disruption budget")
	}
}

// TestEvictionVersions ensures the eviction endpoint accepts and returns the correct API versions
func TestEvictionVersions(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, rm, informers, config, clientSet := rmSetup(tCtx, t)
	defer closeFn()
	defer tCtx.Cancel("test has completed")

	informers.Start(tCtx.Done())
	go rm.Run(tCtx)

	ns := "default"
	subresource := "eviction"
	pod := newPod("test")
	if _, err := clientSet.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}

	podClient := dynamicClient.Resource(schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}).Namespace(ns)

	// get should not be supported
	if _, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{}, subresource); !apierrors.IsMethodNotSupported(err) {
		t.Fatalf("expected MethodNotSupported for GET, got %v", err)
	}

	// patch should not be supported
	for _, patchType := range []types.PatchType{types.JSONPatchType, types.MergePatchType, types.StrategicMergePatchType, types.ApplyPatchType} {
		if _, err := podClient.Patch(context.TODO(), pod.Name, patchType, []byte{}, metav1.PatchOptions{}, subresource); !apierrors.IsMethodNotSupported(err) {
			t.Fatalf("expected MethodNotSupported for GET, got %v", err)
		}
	}

	allowedEvictions := []runtime.Object{
		// v1beta1, no apiVersion/kind
		&policyv1beta1.Eviction{
			TypeMeta:      metav1.TypeMeta{},
			ObjectMeta:    metav1.ObjectMeta{Name: pod.Name},
			DeleteOptions: &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}},
		},
		// v1beta1, apiVersion/kind
		&policyv1beta1.Eviction{
			TypeMeta:      metav1.TypeMeta{APIVersion: "policy/v1beta1", Kind: "Eviction"},
			ObjectMeta:    metav1.ObjectMeta{Name: pod.Name},
			DeleteOptions: &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}},
		},
		// v1, no apiVersion/kind
		&policyv1.Eviction{
			TypeMeta:      metav1.TypeMeta{},
			ObjectMeta:    metav1.ObjectMeta{Name: pod.Name},
			DeleteOptions: &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}},
		},
		// v1, apiVersion/kind
		&policyv1.Eviction{
			TypeMeta:      metav1.TypeMeta{APIVersion: "policy/v1", Kind: "Eviction"},
			ObjectMeta:    metav1.ObjectMeta{Name: pod.Name},
			DeleteOptions: &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}},
		},
	}
	v1Status := schema.GroupVersionKind{Version: "v1", Kind: "Status"}
	for _, allowedEviction := range allowedEvictions {
		data, _ := json.Marshal(allowedEviction)
		u := &unstructured.Unstructured{}
		json.Unmarshal(data, u)
		result, err := podClient.Create(context.TODO(), u, metav1.CreateOptions{}, subresource)
		if err != nil {
			t.Fatalf("error posting %s: %v", string(data), err)
		}
		if result.GroupVersionKind() != v1Status {
			t.Fatalf("expected v1 Status, got %#v", result)
		}
	}

	// create unknown eviction version with apiVersion/kind should fail
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"metadata":   map[string]interface{}{"name": pod.Name},
		"apiVersion": "policy/v2",
		"kind":       "Eviction",
	}}
	if _, err := podClient.Create(context.TODO(), u, metav1.CreateOptions{}, subresource); err == nil {
		t.Fatal("expected error posting unknown Eviction version, got none")
	} else if !strings.Contains(err.Error(), "policy/v2") {
		t.Fatalf("expected error about policy/v2, got %#v", err)
	}
}

// TestEvictionWithFinalizers tests eviction with the use of finalizers
func TestEvictionWithFinalizers(t *testing.T) {
	cases := map[string]struct {
		enablePodDisruptionConditions bool
		phase                         v1.PodPhase
		dryRun                        bool
		wantDisruptionTargetCond      bool
	}{
		"terminal pod with PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodSucceeded,
			wantDisruptionTargetCond:      true,
		},
		"terminal pod with PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			phase:                         v1.PodSucceeded,
			wantDisruptionTargetCond:      false,
		},
		"running pod with PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodRunning,
			wantDisruptionTargetCond:      true,
		},
		"running pod with PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			phase:                         v1.PodRunning,
			wantDisruptionTargetCond:      false,
		},
		"running pod with PodDisruptionConditions enabled should not update conditions in dry-run mode": {
			enablePodDisruptionConditions: true,
			phase:                         v1.PodRunning,
			dryRun:                        true,
			wantDisruptionTargetCond:      false,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			closeFn, rm, informers, _, clientSet := rmSetup(tCtx, t)
			defer closeFn()

			ns := framework.CreateNamespaceOrDie(clientSet, "eviction-with-finalizers", t)
			defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.PodDisruptionConditions, tc.enablePodDisruptionConditions)()
			defer tCtx.Cancel("test has completed")

			informers.Start(tCtx.Done())
			go rm.Run(tCtx)

			pod := newPod("pod")
			pod.ObjectMeta.Finalizers = []string{"test.k8s.io/finalizer"}
			if _, err := clientSet.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create pod: %v", err)
			}

			pod.Status.Phase = tc.phase
			addPodConditionReady(pod)
			if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{}); err != nil {
				t.Fatal(err)
			}

			waitToObservePods(t, informers.Core().V1().Pods().Informer(), 1, tc.phase)
			deleteOption := metav1.DeleteOptions{}
			if tc.dryRun {
				deleteOption.DryRun = []string{metav1.DryRunAll}
			}

			eviction := newV1Eviction(ns.Name, pod.Name, deleteOption)

			err := clientSet.PolicyV1().Evictions(ns.Name).Evict(tCtx, eviction)
			if err != nil {
				t.Fatalf("Eviction of pod failed %v", err)
			}

			updatedPod, e := clientSet.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
			if e != nil {
				t.Fatalf("Failed to get the pod %q with error: %q", klog.KObj(pod), e)
			}
			_, cond := podutil.GetPodCondition(&updatedPod.Status, v1.PodConditionType(v1.DisruptionTarget))
			if tc.wantDisruptionTargetCond && cond == nil {
				t.Errorf("Pod %q does not have the expected condition: %q", klog.KObj(updatedPod), v1.DisruptionTarget)
			} else if !tc.wantDisruptionTargetCond && cond != nil {
				t.Errorf("Pod %q has an unexpected condition: %q", klog.KObj(updatedPod), v1.DisruptionTarget)
			}
		})
	}
}

// TestEvictionWithUnhealthyPodEvictionPolicy tests eviction with a PDB that has a UnhealthyPodEvictionPolicy
func TestEvictionWithUnhealthyPodEvictionPolicy(t *testing.T) {
	cases := map[string]struct {
		enableUnhealthyPodEvictionPolicy bool
		unhealthyPodEvictionPolicy       *policyv1.UnhealthyPodEvictionPolicyType
		isPodReady                       bool
	}{
		"UnhealthyPodEvictionPolicy disabled and policy not set": {
			enableUnhealthyPodEvictionPolicy: false,
			unhealthyPodEvictionPolicy:       nil,
			isPodReady:                       true,
		},
		"UnhealthyPodEvictionPolicy enabled but policy not set": {
			enableUnhealthyPodEvictionPolicy: true,
			unhealthyPodEvictionPolicy:       nil,
			isPodReady:                       true,
		},
		"UnhealthyPodEvictionPolicy enabled but policy set to IfHealthyBudget with ready pod": {
			enableUnhealthyPodEvictionPolicy: true,
			unhealthyPodEvictionPolicy:       unhealthyPolicyPtr(policyv1.IfHealthyBudget),
			isPodReady:                       true,
		},
		"UnhealthyPodEvictionPolicy enabled but policy set to AlwaysAllow with ready pod": {
			enableUnhealthyPodEvictionPolicy: true,
			unhealthyPodEvictionPolicy:       unhealthyPolicyPtr(policyv1.AlwaysAllow),
			isPodReady:                       true,
		},
		"UnhealthyPodEvictionPolicy enabled but policy set to AlwaysAllow with unready pod": {
			enableUnhealthyPodEvictionPolicy: true,
			unhealthyPodEvictionPolicy:       unhealthyPolicyPtr(policyv1.AlwaysAllow),
			isPodReady:                       false,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.PDBUnhealthyPodEvictionPolicy, tc.enableUnhealthyPodEvictionPolicy)()
			closeFn, rm, informers, _, clientSet := rmSetup(tCtx, t)
			defer closeFn()

			ns := framework.CreateNamespaceOrDie(clientSet, "eviction-with-pdb-pod-healthy-policy", t)
			defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
			defer tCtx.Cancel("test has completed")

			informers.Start(tCtx.Done())
			go rm.Run(tCtx)

			pod := newPod("pod")
			if _, err := clientSet.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create pod: %v", err)
			}

			pod.Status.Phase = v1.PodRunning
			if tc.isPodReady {
				addPodConditionReady(pod)
			}

			if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{}); err != nil {
				t.Fatal(err)
			}

			waitToObservePods(t, informers.Core().V1().Pods().Informer(), 1, v1.PodRunning)

			pdb := newPDB()
			pdb.Spec.UnhealthyPodEvictionPolicy = tc.unhealthyPodEvictionPolicy
			if _, err := clientSet.PolicyV1().PodDisruptionBudgets(ns.Name).Create(context.TODO(), pdb, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create PodDisruptionBudget: %v", err)
			}

			if tc.isPodReady {
				waitPDBStable(t, clientSet, ns.Name, pdb.Name, 1)
			} else {
				waitPDB(t, clientSet, ns.Name, pdb.Name, func(pdb *policyv1.PodDisruptionBudget) bool {
					return pdb.Status.ExpectedPods == 1
				})
			}
			// Eviction API can potentially return http.StatusTooManyRequests (429) or http.StatusGatewayTimeout (504) with retryAfterSeconds == 10s
			// Do not retry - we want to test that the first request succeeds and make sure it doesn't unnecessarily block the test for 10s
			policyV1NoRetriesRESTClient := &noRetriesRESTClient{Interface: clientSet.PolicyV1().RESTClient()}
			policyV1NoRetriesClient := policyv1client.New(policyV1NoRetriesRESTClient)

			deleteOption := metav1.DeleteOptions{}
			eviction := newV1Eviction(ns.Name, pod.Name, deleteOption)
			err := policyV1NoRetriesClient.Evictions(ns.Name).Evict(tCtx, eviction)
			if err != nil {
				t.Fatalf("Eviction of pod failed %v", err)
			}
			if policyV1NoRetriesRESTClient.postCalls != 1 {
				t.Fatalf("expected a single POST call, got %d", policyV1NoRetriesRESTClient.postCalls)
			}

			waitToObservePods(t, informers.Core().V1().Pods().Informer(), 0, v1.PodRunning)
			waitPDBStable(t, clientSet, ns.Name, pdb.Name, 0)
		})
	}
}

// TestEvictionWithPrecondition tests eviction with delete preconditions
func TestEvictionWithPrecondition(t *testing.T) {
	cases := map[string]struct {
		enforceResourceVersion     bool
		injectWrongResourceVersion bool
		enforceUID                 bool
		injectWrongUID             bool
		shouldErr                  bool
	}{
		"eviction enforcing resource version": {
			enforceResourceVersion: true,
		},
		"eviction enforcing UID": {
			enforceUID: true,
		},
		"eviction enforcing resource version and UID": {
			enforceUID:             true,
			enforceResourceVersion: true,
		},
		"eviction enforcing wrong resource version should fail": {
			enforceResourceVersion:     true,
			injectWrongResourceVersion: true,
			shouldErr:                  true,
		},
		"eviction enforcing wrong UID should fail": {
			enforceUID:     true,
			injectWrongUID: true,
			shouldErr:      true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			closeFn, rm, informers, _, clientSet := rmSetup(tCtx, t)
			defer closeFn()

			ns := framework.CreateNamespaceOrDie(clientSet, "eviction-with-preconditions", t)
			defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

			defer tCtx.Cancel("test has completed")
			informers.Start(tCtx.Done())
			go rm.Run(tCtx)

			pod := newPod("pod")
			pod, err := clientSet.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
			if err != nil {
				t.Errorf("Failed to create pod: %q", err)
			}

			pod.Status.Phase = v1.PodRunning
			addPodConditionReady(pod)

			// generate a new resource version
			updatedPod, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			waitToObservePods(t, informers.Core().V1().Pods().Informer(), 1, v1.PodRunning)

			deleteOption := metav1.DeleteOptions{}

			if tc.enforceResourceVersion || tc.enforceUID {
				deleteOption.Preconditions = &metav1.Preconditions{}
			}

			if tc.enforceResourceVersion {
				if tc.injectWrongResourceVersion {
					deleteOption.Preconditions.ResourceVersion = &pod.ResourceVersion
				} else {
					deleteOption.Preconditions.ResourceVersion = &updatedPod.ResourceVersion
				}

			}
			if tc.enforceUID {
				if tc.injectWrongUID {
					newUID := uuid.NewUUID()
					deleteOption.Preconditions.UID = &newUID
				} else {
					deleteOption.Preconditions.UID = &updatedPod.UID
				}
			}

			// Eviction API can potentially return http.StatusTooManyRequests (429) or http.StatusGatewayTimeout (504) with retryAfterSeconds == 10s
			// Do not retry - we want to test that the first request succeeds and make sure it doesn't unnecessarily block the test for 10s
			policyV1NoRetriesRESTClient := &noRetriesRESTClient{Interface: clientSet.PolicyV1().RESTClient()}
			policyV1NoRetriesClient := policyv1client.New(policyV1NoRetriesRESTClient)

			eviction := newV1Eviction(ns.Name, updatedPod.Name, deleteOption)
			err = policyV1NoRetriesClient.Evictions(ns.Name).Evict(tCtx, eviction)
			if err != nil && !tc.shouldErr {
				t.Fatalf("Eviction of pod failed %q", err)
			}
			if err == nil && tc.shouldErr {
				t.Fatal("Eviction of pod should fail")
			}
			if policyV1NoRetriesRESTClient.postCalls != 1 {
				t.Fatalf("expected a single POST call, got %d", policyV1NoRetriesRESTClient.postCalls)
			}
		})
	}
}

func newPod(podName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"app": "test-evictions"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
	}
}

func addPodConditionReady(pod *v1.Pod) {
	pod.Status.Conditions = append(pod.Status.Conditions, v1.PodCondition{
		Type:   v1.PodReady,
		Status: v1.ConditionTrue,
	})
}

func newPDB() *policyv1.PodDisruptionBudget {
	return &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pdb",
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			MinAvailable: &intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: 0,
			},
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test-evictions"},
			},
		},
	}
}

func newV1Eviction(ns, evictionName string, deleteOption metav1.DeleteOptions) *policyv1.Eviction {
	return &policyv1.Eviction{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "policy/v1",
			Kind:       "Eviction",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      evictionName,
			Namespace: ns,
		},
		DeleteOptions: &deleteOption,
	}
}

func rmSetup(ctx context.Context, t *testing.T) (kubeapiservertesting.TearDownFunc, *disruption.DisruptionController, informers.SharedInformerFactory, *restclient.Config, clientset.Interface) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "pdb-informers")), resyncPeriod)

	client := clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "disruption-controller"))

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(client.Discovery())
	scaleClient, err := scale.NewForConfig(config, mapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		t.Fatalf("Error in create scaleClient: %v", err)
	}

	rm := disruption.NewDisruptionController(
		ctx,
		informers.Core().V1().Pods(),
		informers.Policy().V1().PodDisruptionBudgets(),
		informers.Core().V1().ReplicationControllers(),
		informers.Apps().V1().ReplicaSets(),
		informers.Apps().V1().Deployments(),
		informers.Apps().V1().StatefulSets(),
		client,
		mapper,
		scaleClient,
		client.Discovery(),
	)
	return server.TearDownFn, rm, informers, config, clientSet
}

// wait for the podInformer to observe the pods. Call this function before
// running the RS controller to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int, phase v1.PodPhase) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) != podNum {
			return false, nil
		}
		for _, obj := range objects {
			pod := obj.(*v1.Pod)
			if pod.Status.Phase != phase {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func waitPDBStable(t *testing.T, clientSet clientset.Interface, ns, pdbName string, podNum int32) {
	waitPDB(t, clientSet, ns, pdbName, func(pdb *policyv1.PodDisruptionBudget) bool {
		return pdb.Status.CurrentHealthy == podNum
	})
}

func waitPDB(t *testing.T, clientSet clientset.Interface, ns, pdbName string, condition func(budget *policyv1.PodDisruptionBudget) bool) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		pdb, err := clientSet.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), pdbName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return condition(pdb), nil
	}); err != nil {
		t.Fatal(err)
	}
}

func unhealthyPolicyPtr(unhealthyPodEvictionPolicy policyv1.UnhealthyPodEvictionPolicyType) *policyv1.UnhealthyPodEvictionPolicyType {
	return &unhealthyPodEvictionPolicy
}

type noRetriesRESTClient struct {
	mu        sync.Mutex
	postCalls int
	restclient.Interface
}

func (n *noRetriesRESTClient) Post() *restclient.Request {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.postCalls++
	return n.Interface.Post().MaxRetries(0)
}
