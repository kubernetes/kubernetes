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

package framework

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	testutils "k8s.io/kubernetes/test/utils"
)

func UpdateDeploymentWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateDeploymentFunc) (*extensions.Deployment, error) {
	return testutils.UpdateDeploymentWithRetries(c, namespace, name, applyUpdate, Logf, Poll, pollShortTimeout)
}

// Waits for the deployment to clean up old rcs.
func WaitForDeploymentOldRSsNum(c clientset.Interface, ns, deploymentName string, desiredRSNum int) error {
	var oldRSs []*extensions.ReplicaSet
	var d *extensions.Deployment

	pollErr := wait.PollImmediate(Poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		d = deployment

		_, oldRSs, err = deploymentutil.GetOldReplicaSets(deployment, c.ExtensionsV1beta1())
		if err != nil {
			return false, err
		}
		return len(oldRSs) == desiredRSNum, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("%d old replica sets were not cleaned up for deployment %q", len(oldRSs)-desiredRSNum, deploymentName)
		logReplicaSetsOfDeployment(d, oldRSs, nil)
	}
	return pollErr
}

func logReplicaSetsOfDeployment(deployment *extensions.Deployment, allOldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet) {
	testutils.LogReplicaSetsOfDeployment(deployment, allOldRSs, newRS, Logf)
}

func WaitForObservedDeployment(c clientset.Interface, ns, deploymentName string, desiredGeneration int64) error {
	return testutils.WaitForObservedDeployment(c, ns, deploymentName, desiredGeneration)
}

func WaitForDeploymentWithCondition(c clientset.Interface, ns, deploymentName, reason string, condType extensions.DeploymentConditionType) error {
	return testutils.WaitForDeploymentWithCondition(c, ns, deploymentName, reason, condType, Logf, Poll, pollLongTimeout)
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly most of the time, but an overwhelmed RS controller
// may result in taking longer to relabel a RS.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string) error {
	return testutils.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, revision, image, Logf, Poll, pollLongTimeout)
}

func NewDeployment(deploymentName string, replicas int32, podLabels map[string]string, imageName, image string, strategyType extensions.DeploymentStrategyType) *extensions.Deployment {
	zero := int64(0)
	return &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			Strategy: extensions.DeploymentStrategy{
				Type: strategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

// Waits for the deployment to complete, and don't check if rolling update strategy is broken.
// Rolling update strategy is used only during a rolling update, and can be violated in other situations,
// such as shortly after a scaling event or the deployment is just created.
func WaitForDeploymentComplete(c clientset.Interface, d *extensions.Deployment) error {
	return testutils.WaitForDeploymentComplete(c, d, Logf, Poll, pollLongTimeout)
}

// Waits for the deployment to complete, and check rolling update strategy isn't broken at any times.
// Rolling update strategy should not be broken during a rolling update.
func WaitForDeploymentCompleteAndCheckRolling(c clientset.Interface, d *extensions.Deployment) error {
	return testutils.WaitForDeploymentCompleteAndCheckRolling(c, d, Logf, Poll, pollLongTimeout)
}

// WaitForDeploymentUpdatedReplicasLTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func WaitForDeploymentUpdatedReplicasLTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int32, desiredGeneration int64) error {
	return testutils.WaitForDeploymentUpdatedReplicasLTE(c, ns, deploymentName, minUpdatedReplicas, desiredGeneration, Poll, pollLongTimeout)
}

// WaitForDeploymentRollbackCleared waits for given deployment either started rolling back or doesn't need to rollback.
// Note that rollback should be cleared shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRollbackCleared(c clientset.Interface, ns, deploymentName string) error {
	return testutils.WaitForDeploymentRollbackCleared(c, ns, deploymentName, Poll, pollShortTimeout)
}

// WatchRecreateDeployment watches Recreate deployments and ensures no new pods will run at the same time with
// old pods.
func WatchRecreateDeployment(c clientset.Interface, d *extensions.Deployment) error {
	if d.Spec.Strategy.Type != extensions.RecreateDeploymentStrategyType {
		return fmt.Errorf("deployment %q does not use a Recreate strategy: %s", d.Name, d.Spec.Strategy.Type)
	}

	w, err := c.Extensions().Deployments(d.Namespace).Watch(metav1.SingleObject(metav1.ObjectMeta{Name: d.Name, ResourceVersion: d.ResourceVersion}))
	if err != nil {
		return err
	}

	status := d.Status

	condition := func(event watch.Event) (bool, error) {
		d := event.Object.(*extensions.Deployment)
		status = d.Status

		if d.Status.UpdatedReplicas > 0 && d.Status.Replicas != d.Status.UpdatedReplicas {
			_, allOldRSs, err := deploymentutil.GetOldReplicaSets(d, c.ExtensionsV1beta1())
			newRS, nerr := deploymentutil.GetNewReplicaSet(d, c.ExtensionsV1beta1())
			if err == nil && nerr == nil {
				Logf("%+v", d)
				logReplicaSetsOfDeployment(d, allOldRSs, newRS)
				logPodsOfDeployment(c, d, append(allOldRSs, newRS))
			}
			return false, fmt.Errorf("deployment %q is running new pods alongside old pods: %#v", d.Name, status)
		}

		return *(d.Spec.Replicas) == d.Status.Replicas &&
			*(d.Spec.Replicas) == d.Status.UpdatedReplicas &&
			d.Generation <= d.Status.ObservedGeneration, nil
	}

	_, err = watch.Until(2*time.Minute, w, condition)
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("deployment %q never completed: %#v", d.Name, status)
	}
	return err
}

func ScaleDeployment(clientset clientset.Interface, internalClientset internalclientset.Interface, ns, name string, size uint, wait bool) error {
	return ScaleResource(clientset, internalClientset, ns, name, size, wait, extensionsinternal.Kind("Deployment"))
}

func RunDeployment(config testutils.DeploymentConfig) error {
	By(fmt.Sprintf("creating deployment %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = DumpNodeDebugInfo
	config.ContainerDumpFunc = LogFailedContainers
	return testutils.RunDeployment(config)
}

func logPodsOfDeployment(c clientset.Interface, deployment *extensions.Deployment, rsList []*extensions.ReplicaSet) {
	testutils.LogPodsOfDeployment(c, deployment, rsList, Logf)
}

func WaitForDeploymentRevision(c clientset.Interface, d *extensions.Deployment, targetRevision string) error {
	err := wait.PollImmediate(Poll, pollLongTimeout, func() (bool, error) {
		deployment, err := c.ExtensionsV1beta1().Deployments(d.Namespace).Get(d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		revision := deployment.Annotations[deploymentutil.RevisionAnnotation]
		return revision == targetRevision, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for revision to become %q for deployment %q: %v", targetRevision, d.Name, err)
	}
	return nil
}

// CheckDeploymentRevisionAndImage checks if the input deployment's and its new replica set's revision and image are as expected.
func CheckDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName, revision, image string) error {
	return testutils.CheckDeploymentRevisionAndImage(c, ns, deploymentName, revision, image)
}
