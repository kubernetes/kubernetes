// +build integration,!no-etcd

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	defaultTimeout = 10 * time.Minute
)

func newPod(podName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
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
		ObjectMeta: v1.ObjectMeta{
			Name: "test-pdb",
		},
		Spec: v1beta1.PodDisruptionBudgetSpec{
			MinAvailable: intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: 0,
			},
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test-evictions"},
			},
		},
	}
}

func newEviction(ns, evictionName string, deleteOption *v1.DeleteOptions) *v1beta1.Eviction {
	return &v1beta1.Eviction{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "Policy/v1beta1",
			Kind:       "Eviction",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      evictionName,
			Namespace: ns,
		},
		DeleteOptions: deleteOption,
	}
}

func rmSetup(t *testing.T) (*httptest.Server, *disruption.DisruptionController, cache.SharedIndexInformer, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pdb-informers")), nil, resyncPeriod)

	rm := disruption.NewDisruptionController(
		informers.Pods().Informer(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "disruption-controller")),
	)
	return s, rm, informers.Pods().Informer(), clientSet
}

func TestConcurrentEvictionRequests(t *testing.T) {
	podNameFormat := "test-pod-%d"

	s, rm, podInformer, clientSet := rmSetup(t)
	defer s.Close()

	ns := framework.CreateTestingNamespace("concurrent-eviction-requests", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	stopCh := make(chan struct{})
	go podInformer.Run(stopCh)
	go rm.Run(stopCh)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)

	var gracePeriodSeconds int64 = 30
	deleteOption := &v1.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}

	// Generate 10 pods to evict
	for i := 0; i < 10; i++ {
		podName := fmt.Sprintf(podNameFormat, i)
		pod := newPod(podName)

		if _, err := clientSet.Core().Pods(ns.Name).Create(pod); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		addPodConditionReady(pod)
		if _, err := clientSet.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
			t.Fatal(err)
		}
	}

	waitToObservePods(t, podInformer, 10)

	pdb := newPDB()
	if _, err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Create(pdb); err != nil {
		t.Errorf("Failed to create PodDisruptionBudget: %v", err)
	}

	waitPDBStable(t, clientSet, 10, ns.Name, pdb.Name)

	doneCh := make(chan bool, 10)
	errCh := make(chan error, 1)
	// spawn 10 goroutine to concurrently evict the pods
	for i := 0; i < 10; i++ {
		go func(id int, doneCh chan bool, errCh chan error) {
			evictionName := fmt.Sprintf(podNameFormat, id)
			eviction := newEviction(ns.Name, evictionName, deleteOption)

			var e error
			for {
				e = clientSet.Policy().Evictions(ns.Name).Evict(eviction)
				if errors.IsTooManyRequests(e) {
					time.Sleep(5 * time.Second)
				} else {
					break
				}
			}
			if e != nil {
				if errors.IsConflict(err) {
					fmt.Errorf("Unexpected Conflict (409) error caused by failing to handle concurrent PDB updates: %v", e)
				} else {
					errCh <- e
				}
				return
			}
			doneCh <- true
		}(i, doneCh, errCh)
	}

	doneCount := 0
	for {
		select {
		case err := <-errCh:
			t.Errorf("%v", err)
			return
		case <-doneCh:
			doneCount++
			if doneCount == 10 {
				return
			}
		case <-time.After(defaultTimeout):
			t.Errorf("Eviction did not complete within %v", defaultTimeout)
		}
	}

	for i := 0; i < 10; i++ {
		podName := fmt.Sprintf(podNameFormat, i)
		_, err := clientSet.Core().Pods(ns.Name).Get(podName, metav1.GetOptions{})
		if !errors.IsNotFound(err) {
			t.Errorf("Pod %q is expected to be evicted", podName)
		}
	}

	if err := clientSet.Policy().PodDisruptionBudgets(ns.Name).Delete(pdb.Name, deleteOption); err != nil {
		t.Errorf("Failed to delete PodDisruptionBudget: %v", err)
	}

	close(stopCh)
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
