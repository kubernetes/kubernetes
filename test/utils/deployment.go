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
	"context"
	"errors"
	"fmt"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

type LogfFn func(format string, args ...interface{})

func LogReplicaSetsOfDeployment(deployment *apps.Deployment, allOldRSs []*apps.ReplicaSet, newRS *apps.ReplicaSet, logf LogfFn) {
	if newRS != nil {
		logf("New ReplicaSet %q of Deployment %q:\n%s", newRS.Name, deployment.Name, dump.Pretty(*newRS))
	} else {
		logf("New ReplicaSet of Deployment %q is nil.", deployment.Name)
	}
	if len(allOldRSs) > 0 {
		logf("All old ReplicaSets of Deployment %q:", deployment.Name)
	}
	for i := range allOldRSs {
		logf(dump.Pretty(*allOldRSs[i]))
	}
}

func LogPodsOfDeployment(c clientset.Interface, deployment *apps.Deployment, rsList []*apps.ReplicaSet, logf LogfFn) {
	minReadySeconds := deployment.Spec.MinReadySeconds
	podListFunc := func(namespace string, options metav1.ListOptions) (*v1.PodList, error) {
		return c.CoreV1().Pods(namespace).List(context.TODO(), options)
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
		logf("Pod %q is %s:\n%s", pod.Name, availability, dump.Pretty(pod))
	}
}

// Waits for the deployment to complete.
// If during a rolling update (rolling == true), returns an error if the deployment's
// rolling update strategy (max unavailable or max surge) is broken at any times.
// It's not seen as a rolling update if shortly after a scaling event or the deployment is just created.
func waitForDeploymentCompleteMaybeCheckRolling(c clientset.Interface, d *apps.Deployment, rolling bool, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var (
		deployment *apps.Deployment
		reason     string
	)

	err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = c.AppsV1().Deployments(d.Namespace).Get(context.TODO(), d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// If during a rolling update, make sure rolling update strategy isn't broken at any times.
		if rolling {
			reason, err = checkRollingUpdateStatus(c, deployment, logf)
			if err != nil {
				return false, err
			}
			logf(reason)
		}

		// When the deployment status and its underlying resources reach the desired state, we're done
		if deploymentutil.DeploymentComplete(d, &deployment.Status) {
			return true, nil
		}

		reason = fmt.Sprintf("deployment status: %#v", deployment.Status)
		logf(reason)

		return false, nil
	})

	if wait.Interrupted(err) {
		err = fmt.Errorf("%s", reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q status to match expectation: %v", d.Name, err)
	}
	return nil
}

func checkRollingUpdateStatus(c clientset.Interface, deployment *apps.Deployment, logf LogfFn) (string, error) {
	var reason string
	oldRSs, allOldRSs, newRS, err := GetAllReplicaSets(deployment, c)
	if err != nil {
		return "", err
	}
	if newRS == nil {
		// New RC hasn't been created yet.
		reason = "new replica set hasn't been created yet"
		return reason, nil
	}
	allRSs := append(oldRSs, newRS)
	// The old/new ReplicaSets need to contain the pod-template-hash label
	for i := range allRSs {
		if !labelsutil.SelectorHasLabel(allRSs[i].Spec.Selector, apps.DefaultDeploymentUniqueLabelKey) {
			reason = "all replica sets need to contain the pod-template-hash label"
			return reason, nil
		}
	}

	// Check max surge and min available
	totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	maxCreated := *(deployment.Spec.Replicas) + deploymentutil.MaxSurge(*deployment)
	if totalCreated > maxCreated {
		LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, logf)
		LogPodsOfDeployment(c, deployment, allRSs, logf)
		return "", fmt.Errorf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
	}
	minAvailable := deploymentutil.MinAvailable(deployment)
	if deployment.Status.AvailableReplicas < minAvailable {
		LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, logf)
		LogPodsOfDeployment(c, deployment, allRSs, logf)
		return "", fmt.Errorf("total pods available: %d, less than the min required: %d", deployment.Status.AvailableReplicas, minAvailable)
	}
	return "", nil
}

// GetAllReplicaSets returns the old and new replica sets targeted by the given Deployment. It gets PodList and ReplicaSetList from client interface.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
// The third returned value is the new replica set, and it may be nil if it doesn't exist yet.
func GetAllReplicaSets(deployment *apps.Deployment, c clientset.Interface) ([]*apps.ReplicaSet, []*apps.ReplicaSet, *apps.ReplicaSet, error) {
	rsList, err := deploymentutil.ListReplicaSets(deployment, deploymentutil.RsListFromClient(c.AppsV1()))
	if err != nil {
		return nil, nil, nil, err
	}
	oldRSes, allOldRSes := deploymentutil.FindOldReplicaSets(deployment, rsList)
	newRS := deploymentutil.FindNewReplicaSet(deployment, rsList)
	return oldRSes, allOldRSes, newRS, nil
}

// GetOldReplicaSets returns the old replica sets targeted by the given Deployment; get PodList and ReplicaSetList from client interface.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func GetOldReplicaSets(deployment *apps.Deployment, c clientset.Interface) ([]*apps.ReplicaSet, []*apps.ReplicaSet, error) {
	rsList, err := deploymentutil.ListReplicaSets(deployment, deploymentutil.RsListFromClient(c.AppsV1()))
	if err != nil {
		return nil, nil, err
	}
	oldRSes, allOldRSes := deploymentutil.FindOldReplicaSets(deployment, rsList)
	return oldRSes, allOldRSes, nil
}

// GetNewReplicaSet returns a replica set that matches the intent of the given deployment; get ReplicaSetList from client interface.
// Returns nil if the new replica set doesn't exist yet.
func GetNewReplicaSet(deployment *apps.Deployment, c clientset.Interface) (*apps.ReplicaSet, error) {
	rsList, err := deploymentutil.ListReplicaSets(deployment, deploymentutil.RsListFromClient(c.AppsV1()))
	if err != nil {
		return nil, err
	}
	return deploymentutil.FindNewReplicaSet(deployment, rsList), nil
}

// Waits for the deployment to complete, and check rolling update strategy isn't broken at any times.
// Rolling update strategy should not be broken during a rolling update.
func WaitForDeploymentCompleteAndCheckRolling(c clientset.Interface, d *apps.Deployment, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	rolling := true
	return waitForDeploymentCompleteMaybeCheckRolling(c, d, rolling, logf, pollInterval, pollTimeout)
}

// Waits for the deployment to complete, and don't check if rolling update strategy is broken.
// Rolling update strategy is used only during a rolling update, and can be violated in other situations,
// such as shortly after a scaling event or the deployment is just created.
func WaitForDeploymentComplete(c clientset.Interface, d *apps.Deployment, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	rolling := false
	return waitForDeploymentCompleteMaybeCheckRolling(c, d, rolling, logf, pollInterval, pollTimeout)
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var deployment *apps.Deployment
	var newRS *apps.ReplicaSet
	var reason string
	err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		deployment, err = c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// The new ReplicaSet needs to be non-nil and contain the pod-template-hash label
		newRS, err = GetNewReplicaSet(deployment, c)
		if err != nil {
			return false, err
		}
		if err := checkRevisionAndImage(deployment, newRS, revision, image); err != nil {
			reason = err.Error()
			logf(reason)
			return false, nil
		}
		return true, nil
	})
	if wait.Interrupted(err) {
		LogReplicaSetsOfDeployment(deployment, nil, newRS, logf)
		err = errors.New(reason)
	}
	if newRS == nil {
		return fmt.Errorf("deployment %q failed to create new replica set", deploymentName)
	}
	if err != nil {
		if deployment == nil {
			return fmt.Errorf("error creating new replica set for deployment %q: %w", deploymentName, err)
		}
		deploymentImage := ""
		if len(deployment.Spec.Template.Spec.Containers) > 0 {
			deploymentImage = deployment.Spec.Template.Spec.Containers[0].Image
		}
		newRSImage := ""
		if len(newRS.Spec.Template.Spec.Containers) > 0 {
			newRSImage = newRS.Spec.Template.Spec.Containers[0].Image
		}
		return fmt.Errorf("error waiting for deployment %q (got %s / %s) and new replica set %q (got %s / %s) revision and image to match expectation (expected %s / %s): %v", deploymentName, deployment.Annotations[deploymentutil.RevisionAnnotation], deploymentImage, newRS.Name, newRS.Annotations[deploymentutil.RevisionAnnotation], newRSImage, revision, image, err)
	}
	return nil
}

// CheckDeploymentRevisionAndImage checks if the input deployment's and its new replica set's revision and image are as expected.
func CheckDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName, revision, image string) error {
	deployment, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("unable to get deployment %s during revision check: %v", deploymentName, err)
	}

	// Check revision of the new replica set of this deployment
	newRS, err := GetNewReplicaSet(deployment, c)
	if err != nil {
		return fmt.Errorf("unable to get new replicaset of deployment %s during revision check: %v", deploymentName, err)
	}
	return checkRevisionAndImage(deployment, newRS, revision, image)
}

func checkRevisionAndImage(deployment *apps.Deployment, newRS *apps.ReplicaSet, revision, image string) error {
	// The new ReplicaSet needs to be non-nil and contain the pod-template-hash label
	if newRS == nil {
		return fmt.Errorf("new replicaset for deployment %q is yet to be created", deployment.Name)
	}
	if !labelsutil.SelectorHasLabel(newRS.Spec.Selector, apps.DefaultDeploymentUniqueLabelKey) {
		return fmt.Errorf("new replica set %q doesn't have %q label selector", newRS.Name, apps.DefaultDeploymentUniqueLabelKey)
	}
	// Check revision of this deployment, and of the new replica set of this deployment
	if deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != revision {
		return fmt.Errorf("deployment %q doesn't have the required revision set", deployment.Name)
	}
	if newRS.Annotations == nil || newRS.Annotations[deploymentutil.RevisionAnnotation] != revision {
		return fmt.Errorf("new replicaset %q doesn't have the required revision set", newRS.Name)
	}
	// Check the image of this deployment, and of the new replica set of this deployment
	if !containsImage(deployment.Spec.Template.Spec.Containers, image) {
		return fmt.Errorf("deployment %q doesn't have the required image %s set", deployment.Name, image)
	}
	if !containsImage(newRS.Spec.Template.Spec.Containers, image) {
		return fmt.Errorf("new replica set %q doesn't have the required image %s.", newRS.Name, image)
	}
	return nil
}

func containsImage(containers []v1.Container, imageName string) bool {
	for _, container := range containers {
		if container.Image == imageName {
			return true
		}
	}
	return false
}

type UpdateDeploymentFunc func(d *apps.Deployment)

func UpdateDeploymentWithRetries(c clientset.Interface, namespace, name string, applyUpdate UpdateDeploymentFunc, logf LogfFn, pollInterval, pollTimeout time.Duration) (*apps.Deployment, error) {
	var deployment *apps.Deployment
	var updateErr error
	pollErr := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		if deployment, err = c.AppsV1().Deployments(namespace).Get(context.TODO(), name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(deployment)
		if deployment, err = c.AppsV1().Deployments(namespace).Update(context.TODO(), deployment, metav1.UpdateOptions{}); err == nil {
			logf("Updating deployment %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if wait.Interrupted(pollErr) {
		pollErr = fmt.Errorf("couldn't apply the provided updated to deployment %q: %v", name, updateErr)
	}
	return deployment, pollErr
}

func WaitForObservedDeployment(c clientset.Interface, ns, deploymentName string, desiredGeneration int64) error {
	return deploymentutil.WaitForObservedDeployment(func() (*apps.Deployment, error) {
		return c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	}, desiredGeneration, 2*time.Second, 1*time.Minute)
}

// WaitForDeploymentUpdatedReplicasGTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func WaitForDeploymentUpdatedReplicasGTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int32, desiredGeneration int64, pollInterval, pollTimeout time.Duration) error {
	var deployment *apps.Deployment
	err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		d, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		deployment = d
		return deployment.Status.ObservedGeneration >= desiredGeneration && deployment.Status.UpdatedReplicas >= minUpdatedReplicas, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q to have at least %d updatedReplicas: %v; latest .status.updatedReplicas: %d", deploymentName, minUpdatedReplicas, err, deployment.Status.UpdatedReplicas)
	}
	return nil
}

func WaitForDeploymentWithCondition(c clientset.Interface, ns, deploymentName, reason string, condType apps.DeploymentConditionType, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var deployment *apps.Deployment
	pollErr := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		d, err := c.AppsV1().Deployments(ns).Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		deployment = d
		cond := deploymentutil.GetDeploymentCondition(deployment.Status, condType)
		return cond != nil && cond.Reason == reason, nil
	})
	if wait.Interrupted(pollErr) {
		pollErr = fmt.Errorf("deployment %q never updated with the desired condition and reason, latest deployment conditions: %+v", deployment.Name, deployment.Status.Conditions)
		_, allOldRSs, newRS, err := GetAllReplicaSets(deployment, c)
		if err == nil {
			LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, logf)
			LogPodsOfDeployment(c, deployment, append(allOldRSs, newRS), logf)
		}
	}
	return pollErr
}
