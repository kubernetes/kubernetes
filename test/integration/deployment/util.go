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

package deployment

import (
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/deployment"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
)

const (
	pollInterval = 100 * time.Millisecond
	pollTimeout  = 60 * time.Second

	fakeImageName = "fake-name"
	fakeImage     = "fakeimage"
)

var pauseFn = func(update *v1beta1.Deployment) {
	update.Spec.Paused = true
}

var resumeFn = func(update *v1beta1.Deployment) {
	update.Spec.Paused = false
}

type deploymentTester struct {
	t          *testing.T
	c          clientset.Interface
	deployment *v1beta1.Deployment
}

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

// newDeployment returns a RollingUpdate Deployment with with a fake container image
func newDeployment(name, ns string, replicas int32) *v1beta1.Deployment {
	return &v1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: testLabels()},
			Strategy: v1beta1.DeploymentStrategy{
				Type: v1beta1.RollingUpdateDeploymentStrategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: testLabels(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  fakeImageName,
							Image: fakeImage,
						},
					},
				},
			},
		},
	}
}

// dcSetup sets up necessities for Deployment integration test, including master, apiserver, informers, and clientset
func dcSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *replicaset.ReplicaSetController, *deployment.DeploymentController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "deployment-informers")), resyncPeriod)

	dc := deployment.NewDeploymentController(
		informers.Extensions().V1beta1().Deployments(),
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Core().V1().Pods(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "deployment-controller")),
	)
	rm := replicaset.NewReplicaSetController(
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Core().V1().Pods(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "replicaset-controller")),
		replicaset.BurstReplicas,
	)
	return s, closeFn, rm, dc, informers, clientSet
}

// dcSimpleSetup sets up necessities for Deployment integration test, including master, apiserver,
// and clientset, but not controllers and informers
func dcSimpleSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	return s, closeFn, clientSet
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

func (d *deploymentTester) waitForDeploymentRevisionAndImage(revision, image string) error {
	if err := testutil.WaitForDeploymentRevisionAndImage(d.c, d.deployment.Namespace, d.deployment.Name, revision, image, d.t.Logf, pollInterval, pollTimeout); err != nil {
		return fmt.Errorf("failed to wait for Deployment revision %s: %v", d.deployment.Name, err)
	}
	return nil
}

// markAllPodsReady manually updates all Deployment pods status to ready
func (d *deploymentTester) markAllPodsReady() {
	ns := d.deployment.Namespace
	selector, err := metav1.LabelSelectorAsSelector(d.deployment.Spec.Selector)
	if err != nil {
		d.t.Fatalf("failed to parse Deployment selector: %v", err)
	}
	var readyPods int32
	err = wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		readyPods = 0
		pods, err := d.c.Core().Pods(ns).List(metav1.ListOptions{LabelSelector: selector.String()})
		if err != nil {
			d.t.Logf("failed to list Deployment pods, will retry later: %v", err)
			return false, nil
		}
		for i := range pods.Items {
			pod := pods.Items[i]
			if podutil.IsPodReady(&pod) {
				readyPods++
				continue
			}
			addPodConditionReady(&pod, metav1.Now())
			if _, err = d.c.Core().Pods(ns).UpdateStatus(&pod); err != nil {
				d.t.Logf("failed to update Deployment pod %s, will retry later: %v", pod.Name, err)
			} else {
				readyPods++
			}
		}
		if readyPods >= *d.deployment.Spec.Replicas {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		d.t.Fatalf("failed to mark all Deployment pods to ready: %v", err)
	}
}

func (d *deploymentTester) waitForDeploymentStatusValid() error {
	return testutil.WaitForDeploymentStatusValid(d.c, d.deployment, d.t.Logf, pollInterval, pollTimeout)
}

// waitForDeploymentStatusValidAndMarkPodsReady waits for the Deployment status to become valid
// while marking all Deployment pods as ready at the same time.
func (d *deploymentTester) waitForDeploymentStatusValidAndMarkPodsReady() error {
	// Manually mark all Deployment pods as ready in a separate goroutine
	go d.markAllPodsReady()

	// Make sure the Deployment status is valid while Deployment pods are becoming ready
	err := d.waitForDeploymentStatusValid()
	if err != nil {
		return fmt.Errorf("failed to wait for Deployment status %s: %v", d.deployment.Name, err)
	}
	return nil
}

func (d *deploymentTester) updateDeployment(applyUpdate testutil.UpdateDeploymentFunc) (*v1beta1.Deployment, error) {
	return testutil.UpdateDeploymentWithRetries(d.c, d.deployment.Namespace, d.deployment.Name, applyUpdate, d.t.Logf, pollInterval, pollTimeout)
}

func (d *deploymentTester) waitForObservedDeployment(desiredGeneration int64) error {
	if err := testutil.WaitForObservedDeployment(d.c, d.deployment.Namespace, d.deployment.Name, desiredGeneration); err != nil {
		return fmt.Errorf("failed waiting for ObservedGeneration of deployment %s to become %d: %v", d.deployment.Name, desiredGeneration, err)
	}
	return nil
}

func (d *deploymentTester) getNewReplicaSet() (*v1beta1.ReplicaSet, error) {
	rs, err := deploymentutil.GetNewReplicaSet(d.deployment, d.c.ExtensionsV1beta1())
	if err != nil {
		return nil, fmt.Errorf("failed retrieving new replicaset of deployment %s: %v", d.deployment.Name, err)
	}
	return rs, nil
}

func (d *deploymentTester) expectNoNewReplicaSet() error {
	rs, err := d.getNewReplicaSet()
	if err != nil {
		return err
	}
	if rs != nil {
		return fmt.Errorf("expected deployment %s not to create a new replicaset, got %v", d.deployment.Name, rs)
	}
	return nil
}

func (d *deploymentTester) expectNewReplicaSet() (*v1beta1.ReplicaSet, error) {
	rs, err := d.getNewReplicaSet()
	if err != nil {
		return nil, err
	}
	if rs == nil {
		return nil, fmt.Errorf("expected deployment %s to create a new replicaset, got nil", d.deployment.Name)
	}
	return rs, nil
}

func (d *deploymentTester) updateReplicaSet(name string, applyUpdate testutil.UpdateReplicaSetFunc) (*v1beta1.ReplicaSet, error) {
	return testutil.UpdateReplicaSetWithRetries(d.c, d.deployment.Namespace, name, applyUpdate, d.t.Logf, pollInterval, pollTimeout)
}
