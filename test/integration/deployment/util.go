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

	"github.com/davecgh/go-spew/spew"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller/deployment"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/test/integration/framework"
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

func (d *deploymentTester) logReplicaSetsOfDeployment(allOldRSs []*v1beta1.ReplicaSet, newRS *v1beta1.ReplicaSet) {
	if newRS != nil {
		d.t.Logf("New ReplicaSet of Deployment %s:\n%+v", d.deployment.Name, *newRS)
	} else {
		d.t.Logf("New ReplicaSet of Deployment %s is nil.", d.deployment.Name)
	}
	if len(allOldRSs) > 0 {
		d.t.Logf("All old ReplicaSets of Deployment %s:", d.deployment.Name)
	}
	for i := range allOldRSs {
		d.t.Logf(spew.Sprintf("%#v", *allOldRSs[i]))
	}
}

func (d *deploymentTester) logPodsOfDeployment(rsList []*v1beta1.ReplicaSet) {
	minReadySeconds := d.deployment.Spec.MinReadySeconds
	podListFunc := func(namespace string, options metav1.ListOptions) (*v1.PodList, error) {
		return d.c.Core().Pods(namespace).List(options)
	}

	podList, err := deploymentutil.ListPods(d.deployment, rsList, podListFunc)

	if err != nil {
		d.t.Logf("Failed to list Pods of Deployment %s: %v", d.deployment.Name, err)
		return
	}
	for _, pod := range podList.Items {
		availability := "not available"
		if podutil.IsPodAvailable(&pod, minReadySeconds, metav1.Now()) {
			availability = "available"
		}
		d.t.Logf("Pod %s is %s:\n%s", pod.Name, availability, spew.Sprintf("%#v", pod))
	}
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly, so we only wait for 1 minute here to fail early.
func (d *deploymentTester) waitForDeploymentRevisionAndImage(revision, image string) error {
	var deployment *v1beta1.Deployment
	var newRS *v1beta1.ReplicaSet
	var reason string
	deploymentName, ns := d.deployment.Name, d.deployment.Namespace
	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = d.c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// The new ReplicaSet needs to be non-nil and contain the pod-template-hash label
		newRS, err = deploymentutil.GetNewReplicaSet(deployment, d.c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			reason = fmt.Sprintf("New replica set for deployment %q is yet to be created", deployment.Name)
			d.t.Logf(reason)
			return false, nil
		}
		if !labelsutil.SelectorHasLabel(newRS.Spec.Selector, v1beta1.DefaultDeploymentUniqueLabelKey) {
			reason = fmt.Sprintf("New replica set %q doesn't have DefaultDeploymentUniqueLabelKey", newRS.Name)
			d.t.Logf(reason)
			return false, nil
		}
		// Check revision of this deployment, and of the new replica set of this deployment
		if deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != revision {
			reason = fmt.Sprintf("Deployment %q doesn't have the required revision set", deployment.Name)
			d.t.Logf(reason)
			return false, nil
		}
		if deployment.Spec.Template.Spec.Containers[0].Image != image {
			reason = fmt.Sprintf("Deployment %q doesn't have the required image set", deployment.Name)
			d.t.Logf(reason)
			return false, nil
		}
		if newRS.Annotations == nil || newRS.Annotations[deploymentutil.RevisionAnnotation] != revision {
			reason = fmt.Sprintf("New replica set %q doesn't have the required revision set", newRS.Name)
			d.t.Logf(reason)
			return false, nil
		}
		if newRS.Spec.Template.Spec.Containers[0].Image != image {
			reason = fmt.Sprintf("New replica set %q doesn't have the required image set", newRS.Name)
			d.t.Logf(reason)
			return false, nil
		}
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		d.logReplicaSetsOfDeployment(nil, newRS)
		err = fmt.Errorf(reason)
	}
	if newRS == nil {
		return fmt.Errorf("deployment %q failed to create new replica set", deploymentName)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q (got %s / %s) and new replica set %q (got %s / %s) revision and image to match expectation (expected %s / %s): %v", deploymentName, deployment.Annotations[deploymentutil.RevisionAnnotation], deployment.Spec.Template.Spec.Containers[0].Image, newRS.Name, newRS.Annotations[deploymentutil.RevisionAnnotation], newRS.Spec.Template.Spec.Containers[0].Image, revision, image, err)
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
	var (
		oldRSs, allOldRSs, allRSs []*v1beta1.ReplicaSet
		newRS                     *v1beta1.ReplicaSet
		deployment                *v1beta1.Deployment
		reason                    string
	)

	name := d.deployment.Name
	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = d.c.Extensions().Deployments(d.deployment.Namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		oldRSs, allOldRSs, newRS, err = deploymentutil.GetAllReplicaSets(deployment, d.c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			// New RC hasn't been created yet.
			reason = "new replica set hasn't been created yet"
			d.t.Logf(reason)
			return false, nil
		}
		allRSs = append(oldRSs, newRS)
		// The old/new ReplicaSets need to contain the pod-template-hash label
		for i := range allRSs {
			if !labelsutil.SelectorHasLabel(allRSs[i].Spec.Selector, v1beta1.DefaultDeploymentUniqueLabelKey) {
				reason = "all replica sets need to contain the pod-template-hash label"
				d.t.Logf(reason)
				return false, nil
			}
		}
		totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
		maxCreated := *(deployment.Spec.Replicas) + deploymentutil.MaxSurge(*deployment)
		if totalCreated > maxCreated {
			reason = fmt.Sprintf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
			d.t.Logf(reason)
			return false, nil
		}
		minAvailable := deploymentutil.MinAvailable(deployment)
		if deployment.Status.AvailableReplicas < minAvailable {
			reason = fmt.Sprintf("total pods available: %d, less than the min required: %d", deployment.Status.AvailableReplicas, minAvailable)
			d.t.Logf(reason)
			return false, nil
		}

		// When the deployment status and its underlying resources reach the desired state, we're done
		if deploymentutil.DeploymentComplete(deployment, &deployment.Status) {
			return true, nil
		}

		reason = fmt.Sprintf("deployment status: %#v", deployment.Status)
		d.t.Logf(reason)

		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		d.logReplicaSetsOfDeployment(allOldRSs, newRS)
		d.logPodsOfDeployment(allRSs)
		err = fmt.Errorf("%s", reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q status to match expectation: %v", d.deployment.Name, err)
	}
	return nil
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
