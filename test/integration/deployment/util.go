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
	"net/http/httptest"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
)

const (
	pollInterval = 1 * time.Second
	pollTimeout  = 60 * time.Second

	fakeImageName = "fake-name"
	fakeImage     = "fakeimage"
)

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
	return testutil.WaitForDeploymentRevisionAndImage(d.c, d.deployment.Namespace, d.deployment.Name, revision, image, d.t.Logf, pollInterval, pollTimeout)
}

// markAllPodsReady manually updates all Deployment pods status to ready
func (d *deploymentTester) markAllPodsReady() {
	ns := d.deployment.Namespace
	selector, err := metav1.LabelSelectorAsSelector(d.deployment.Spec.Selector)
	if err != nil {
		d.t.Fatalf("failed to parse Deployment selector: %v", err)
	}
	var readyPods int32
	err = wait.Poll(100*time.Millisecond, pollTimeout, func() (bool, error) {
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
func (d *deploymentTester) waitForDeploymentStatusValidAndMarkPodsReady() {
	// Manually mark all Deployment pods as ready in a separate goroutine
	go d.markAllPodsReady()

	// Make sure the Deployment status is valid while Deployment pods are becoming ready
	err := d.waitForDeploymentStatusValid()
	if err != nil {
		d.t.Fatalf("failed to wait for Deployment status %s: %v", d.deployment.Name, err)
	}
}
