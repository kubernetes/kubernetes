/*
Copyright 2017 The Kubernetes Authors.

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
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/apps/v1beta2"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/statefulset"
	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
)

const (
	pollInterval = 1 * time.Second
	pollTimeout  = 60 * time.Second

	fakeImageName = "fake-name"
	fakeImage     = "fakeimage"
)

type statefulsetTester struct {
	t           *testing.T
	c           clientset.Interface
	service     *api.Service
	statefulset *v1beta2.StatefulSet
}

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

// newService returns a service with a fake name for StatefulSet to be created soon
func newService(namespace string) *api.Service {
	return &api.Service{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Service",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      "fake-service-name",
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{NodePort: 111},
			},
		},
	}
}

// newStatefulSet returns a StatefulSet with with a fake container image
func newStatefulset(name, namespace string, replicas int) *v1beta2.StatefulSet {
	replicasCopy := int32(replicas)
	return &v1beta2.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1beta2",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1beta2.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: testLabels(),
			},
			Replicas: &replicasCopy,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: testLabels(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			},
			ServiceName: "fake-service-name",
			UpdateStrategy: v1beta2.StatefulSetUpdateStrategy{
				Type: v1beta2.OnDeleteStatefulSetStrategyType,
			},
		},
	}
}

func newMatchingPod(podName, namespace string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels:    testLabels(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

// scSetup sets up necessities for Statefulset integration test, including master, apiserver, informers, and clientset
func scSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *statefulset.StatefulSetController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "statefulset-informers")), resyncPeriod)

	sc := statefulset.NewStatefulSetController(
		informers.Core().V1().Pods(),
		informers.Apps().V1beta1().StatefulSets(), //TODO: use v1beta2 informer when v1beta2 is enabled
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Apps().V1beta1().ControllerRevisions(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "statefulset-controller")),
	)

	return s, closeFn, sc, informers, clientSet
}

// addPodConditionReady sets given pod status to ready at given time
func addPodConditionReady(pod *v1.Pod, time metav1.Time) {
	pod.Status = v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:               v1.PodReady,
				Status:             v1.ConditionTrue,
				LastTransitionTime: time,
			},
		},
	}
}

func (s *statefulsetTester) waitForStatefulSetImage(revision, image string) error {
	return testutil.WaitForStatefulSetImage(s.c, s.statefulset.Namespace, s.statefulset.Name, revision, image, s.t.Logf, pollInterval, pollTimeout)
}

// markAllPodsReady manually updates all Statefulset pods status to ready
func (s *statefulsetTester) markAllPodsReady() {
	ns := s.statefulset.Namespace
	selector, err := metav1.LabelSelectorAsSelector(s.statefulset.Spec.Selector)
	if err != nil {
		s.t.Fatalf("failed to parse Statefulset selector: %v", err)
	}
	var readyPods int32
	err = wait.Poll(100*time.Millisecond, pollTimeout, func() (bool, error) {
		readyPods = 0
		pods, err := s.c.Core().Pods(ns).List(metav1.ListOptions{LabelSelector: selector.String()})
		if err != nil {
			s.t.Logf("failed to list Statefulset pods, will retry later: %v", err)
			return false, nil
		}
		for i := range pods.Items {
			pod := pods.Items[i]
			if podutil.IsPodReady(&pod) {
				readyPods++
				continue
			}
			addPodConditionReady(&pod, metav1.Now())
			if _, err = s.c.Core().Pods(ns).UpdateStatus(&pod); err != nil {
				s.t.Logf("failed to update Statefulset pod %s, will retry later: %v", pod.Name, err)
			} else {
				readyPods++
			}
		}
		if readyPods >= *s.statefulset.Spec.Replicas {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		s.t.Fatalf("failed to mark all Statefulset pods to ready: %v", err)
	}
}

func (s *statefulsetTester) waitForStatefulSetStatusValid() error {
	return testutil.WaitForStatefulSetStatusValid(s.c, s.statefulset, s.t.Logf, pollInterval, pollTimeout)
}

// waitForStatefulSetStatusValidAndMarkPodsReady waits for the StatefulSet status to become valid
// while marking all StatefulSet pods as ready at the same time.
func (s *statefulsetTester) waitForStatefulSetStatusValidAndMarkPodsReady() {
	// Manually mark all StatefulSet pods as ready in a separate goroutine
	go s.markAllPodsReady()

	// Make sure the StatefulSet status is valid while StatefulSet pods are becoming ready
	err := s.waitForStatefulSetStatusValid()
	if err != nil {
		s.t.Fatalf("failed to wait for StatefulSet status %s: %v", s.statefulset.Name, err)
	}
}
