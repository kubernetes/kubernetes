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
	"fmt"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/test/integration/framework"
	"reflect"
)

const (
	numOfEvictions = 10
)

// TestConcurrentEvictionRequests is to make sure pod disruption budgets (PDB) controller is able to
// handle concurrent eviction requests. Original issue:#37605
func TestConcurrentEvictionRequests(t *testing.T) {
	podNameFormat := "test-pod-%d"

	s, closeFn, rm, informers, clientSet := rmSetup(t)
	defer closeFn()

	ns := framework.CreateTestingNamespace("concurrent-eviction-requests", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go rm.Run(stopCh)
	defer close(stopCh)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}

	var gracePeriodSeconds int64 = 30
	deleteOption := &metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}

	// Generate numOfEvictions pods to evict
	for i := 0; i < numOfEvictions; i++ {
		podName := fmt.Sprintf(podNameFormat, i)
		pod := newPod(podName)

		if _, err := clientSet.CoreV1().Pods(ns.Name).Create(pod); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		addPodConditionReady(pod)
		if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(pod); err != nil {
			t.Fatal(err)
		}
	}

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), numOfEvictions)

	pdb := newPDB()
	if _, err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Create(pdb); err != nil {
		t.Errorf("Failed to create PodDisruptionBudget: %v", err)
	}

	waitPDBStable(t, clientSet, numOfEvictions, ns.Name, pdb.Name)

	var numberPodsEvicted uint32 = 0
	errCh := make(chan error, 3*numOfEvictions)
	var wg sync.WaitGroup
	// spawn numOfEvictions goroutines to concurrently evict the pods
	for i := 0; i < numOfEvictions; i++ {
		wg.Add(1)
		go func(id int, errCh chan error) {
			defer wg.Done()
			podName := fmt.Sprintf(podNameFormat, id)
			eviction := newEviction(ns.Name, podName, deleteOption)

			err := wait.PollImmediate(5*time.Second, 60*time.Second, func() (bool, error) {
				e := clientSet.Policy().Evictions(ns.Name).Evict(eviction)
				switch {
				case errors.IsTooManyRequests(e):
					return false, nil
				case errors.IsConflict(e):
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

			_, err = clientSet.CoreV1().Pods(ns.Name).Get(podName, metav1.GetOptions{})
			switch {
			case errors.IsNotFound(err):
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
			e := clientSet.CoreV1().Pods(ns.Name).Delete(podName, deleteOption)
			if e != nil {
				errCh <- e
			}

		}(i, errCh)
	}

	wg.Wait()

	close(errCh)
	var errList []error
	if err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Delete(pdb.Name, deleteOption); err != nil {
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
	s, closeFn, rm, informers, clientSet := rmSetup(t)
	defer closeFn()

	ns := framework.CreateTestingNamespace("terminalpod-eviction", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go rm.Run(stopCh)
	defer close(stopCh)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Failed to create clientset: %v", err)
	}

	var gracePeriodSeconds int64 = 30
	deleteOption := &metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}
	pod := newPod("test-terminal-pod1")
	if _, err := clientSet.CoreV1().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	addPodConditionSucceeded(pod)
	if _, err := clientSet.CoreV1().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Fatal(err)
	}

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), 1)

	pdb := newPDB()
	if _, err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Create(pdb); err != nil {
		t.Errorf("Failed to create PodDisruptionBudget: %v", err)
	}

	pdbList, err := clientSet.Policy().PodDisruptionBudgets(ns.Name).List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Error while listing pod disruption budget")
	}
	oldPdb := pdbList.Items[0]
	eviction := newEviction(ns.Name, pod.Name, deleteOption)
	err = wait.PollImmediate(5*time.Second, 60*time.Second, func() (bool, error) {
		e := clientSet.Policy().Evictions(ns.Name).Evict(eviction)
		switch {
		case errors.IsTooManyRequests(e):
			return false, nil
		case errors.IsConflict(e):
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
	pdbList, err = clientSet.Policy().PodDisruptionBudgets(ns.Name).List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Error while listing pod disruption budget")
	}
	newPdb := pdbList.Items[0]
	// We shouldn't see an update in pod disruption budget status' generation number as we are evicting terminal pods without checking for pod disruption.
	if !reflect.DeepEqual(newPdb.Status.ObservedGeneration, oldPdb.Status.ObservedGeneration) {
		t.Fatalf("Expected the pdb generation to be of same value %v but got %v", newPdb.Status.ObservedGeneration, oldPdb.Status.ObservedGeneration)
	}

	if err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Delete(pdb.Name, deleteOption); err != nil {
		t.Fatalf("Failed to delete pod disruption budget")
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

func addPodConditionSucceeded(pod *v1.Pod) {
	pod.Status = v1.PodStatus{
		Phase: v1.PodSucceeded,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
		},
	}
}

func addPodConditionReady(pod *v1.Pod) {
	pod.Status = v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
		},
	}
}

func newPDB() *v1beta1.PodDisruptionBudget {
	return &v1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pdb",
		},
		Spec: v1beta1.PodDisruptionBudgetSpec{
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

func newEviction(ns, evictionName string, deleteOption *metav1.DeleteOptions) *v1beta1.Eviction {
	return &v1beta1.Eviction{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "Policy/v1beta1",
			Kind:       "Eviction",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      evictionName,
			Namespace: ns,
		},
		DeleteOptions: deleteOption,
	}
}

func rmSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *disruption.DisruptionController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pdb-informers")), resyncPeriod)

	rm := disruption.NewDisruptionController(
		informers.Core().V1().Pods(),
		informers.Policy().V1beta1().PodDisruptionBudgets(),
		informers.Core().V1().ReplicationControllers(),
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Extensions().V1beta1().Deployments(),
		informers.Apps().V1beta1().StatefulSets(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "disruption-controller")),
	)
	return s, closeFn, rm, informers, clientSet
}

// wait for the podInformer to observe the pods. Call this function before
// running the RS controller to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) == podNum {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func waitPDBStable(t *testing.T, clientSet clientset.Interface, podNum int32, ns, pdbName string) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		pdb, err := clientSet.Policy().PodDisruptionBudgets(ns).Get(pdbName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.CurrentHealthy != podNum {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}
