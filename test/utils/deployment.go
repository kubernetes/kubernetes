/*
Copyright 2016 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"time"

	"github.com/davecgh/go-spew/spew"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

type LogfFn func(format string, args ...interface{})

func LogReplicaSetsOfDeployment(deployment *extensions.Deployment, allOldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, logf LogfFn) {
	if newRS != nil {
		logf(spew.Sprintf("New ReplicaSet %q of Deployment %q:\n%+v", newRS.Name, deployment.Name, *newRS))
	} else {
		logf("New ReplicaSet of Deployment %q is nil.", deployment.Name)
	}
	if len(allOldRSs) > 0 {
		logf("All old ReplicaSets of Deployment %q:", deployment.Name)
	}
	for i := range allOldRSs {
		logf(spew.Sprintf("%+v", *allOldRSs[i]))
	}
}

func LogPodsOfDeployment(c clientset.Interface, deployment *extensions.Deployment, rsList []*extensions.ReplicaSet, logf LogfFn) {
	minReadySeconds := deployment.Spec.MinReadySeconds
	podListFunc := func(namespace string, options metav1.ListOptions) (*v1.PodList, error) {
		return c.Core().Pods(namespace).List(options)
	}

	podList, err := deploymentutil.ListPods(deployment, rsList, podListFunc)
	if err != nil {
		logf("Failed to list Pods of Deployment %q: %v", deployment.Name, err)
		return
	}
	for _, pod := range podList.Items {
		availability := "not available"
		if podutil.IsPodAvailable(&pod, minReadySeconds, metav1.Now()) {
			availability = "available"
		}
		logf(spew.Sprintf("Pod %q is %s:\n%+v", pod.Name, availability, pod))
	}
}

// Waits for the deployment status to become valid (i.e. max unavailable and max surge aren't violated anymore).
// Note that the status should stay valid at all times unless shortly after a scaling event or the deployment is just created.
// To verify that the deployment status is valid and wait for the rollout to finish, use WaitForDeploymentStatus instead.
func WaitForDeploymentStatusValid(c clientset.Interface, d *extensions.Deployment, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var (
		oldRSs, allOldRSs, allRSs []*extensions.ReplicaSet
		newRS                     *extensions.ReplicaSet
		deployment                *extensions.Deployment
		reason                    string
	)

	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = c.Extensions().Deployments(d.Namespace).Get(d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		oldRSs, allOldRSs, newRS, err = deploymentutil.GetAllReplicaSets(deployment, c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			// New RC hasn't been created yet.
			reason = "new replica set hasn't been created yet"
			logf(reason)
			return false, nil
		}
		allRSs = append(oldRSs, newRS)
		// The old/new ReplicaSets need to contain the pod-template-hash label
		for i := range allRSs {
			if !labelsutil.SelectorHasLabel(allRSs[i].Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
				reason = "all replica sets need to contain the pod-template-hash label"
				logf(reason)
				return false, nil
			}
		}
		totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
		maxCreated := *(deployment.Spec.Replicas) + deploymentutil.MaxSurge(*deployment)
		if totalCreated > maxCreated {
			reason = fmt.Sprintf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
			logf(reason)
			return false, nil
		}
		minAvailable := deploymentutil.MinAvailable(deployment)
		if deployment.Status.AvailableReplicas < minAvailable {
			reason = fmt.Sprintf("total pods available: %d, less than the min required: %d", deployment.Status.AvailableReplicas, minAvailable)
			logf(reason)
			return false, nil
		}

		// When the deployment status and its underlying resources reach the desired state, we're done
		if deploymentutil.DeploymentComplete(deployment, &deployment.Status) {
			return true, nil
		}

		reason = fmt.Sprintf("deployment status: %#v", deployment.Status)
		logf(reason)

		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, logf)
		LogPodsOfDeployment(c, deployment, allRSs, logf)
		err = fmt.Errorf("%s", reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q status to match expectation: %v", d.Name, err)
	}
	return nil
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var deployment *extensions.Deployment
	var newRS *extensions.ReplicaSet
	var reason string
	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// The new ReplicaSet needs to be non-nil and contain the pod-template-hash label

		newRS, err = deploymentutil.GetNewReplicaSet(deployment, c)

		if err != nil {
			return false, err
		}
		if newRS == nil {
			reason = fmt.Sprintf("New replica set for deployment %q is yet to be created", deployment.Name)
			logf(reason)
			return false, nil
		}
		if !labelsutil.SelectorHasLabel(newRS.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
			reason = fmt.Sprintf("New replica set %q doesn't have DefaultDeploymentUniqueLabelKey", newRS.Name)
			logf(reason)
			return false, nil
		}
		// Check revision of this deployment, and of the new replica set of this deployment
		if deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != revision {
			reason = fmt.Sprintf("Deployment %q doesn't have the required revision set", deployment.Name)
			logf(reason)
			return false, nil
		}
		if deployment.Spec.Template.Spec.Containers[0].Image != image {
			reason = fmt.Sprintf("Deployment %q doesn't have the required image set", deployment.Name)
			logf(reason)
			return false, nil
		}
		if newRS.Annotations == nil || newRS.Annotations[deploymentutil.RevisionAnnotation] != revision {
			reason = fmt.Sprintf("New replica set %q doesn't have the required revision set", newRS.Name)
			logf(reason)
			return false, nil
		}
		if newRS.Spec.Template.Spec.Containers[0].Image != image {
			reason = fmt.Sprintf("New replica set %q doesn't have the required image set", newRS.Name)
			logf(reason)
			return false, nil
		}
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		LogReplicaSetsOfDeployment(deployment, nil, newRS, logf)
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
