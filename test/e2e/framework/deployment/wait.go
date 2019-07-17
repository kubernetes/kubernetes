/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// poll is how often to poll pods, nodes and claims.
	poll             = 2 * time.Second
	pollShortTimeout = 1 * time.Minute
	pollLongTimeout  = 5 * time.Minute
)

// WaitForObservedDeployment waits for the specified deployment generation.
func WaitForObservedDeployment(c clientset.Interface, ns, deploymentName string, desiredGeneration int64) error {
	return testutils.WaitForObservedDeployment(c, ns, deploymentName, desiredGeneration)
}

// WaitForDeploymentWithCondition waits for the specified deployment condition.
func WaitForDeploymentWithCondition(c clientset.Interface, ns, deploymentName, reason string, condType appsv1.DeploymentConditionType) error {
	return testutils.WaitForDeploymentWithCondition(c, ns, deploymentName, reason, condType, e2elog.Logf, poll, pollLongTimeout)
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly most of the time, but an overwhelmed RS controller
// may result in taking longer to relabel a RS.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string) error {
	return testutils.WaitForDeploymentRevisionAndImage(c, ns, deploymentName, revision, image, e2elog.Logf, poll, pollLongTimeout)
}

// WaitForDeploymentComplete waits for the deployment to complete, and don't check if rolling update strategy is broken.
// Rolling update strategy is used only during a rolling update, and can be violated in other situations,
// such as shortly after a scaling event or the deployment is just created.
func WaitForDeploymentComplete(c clientset.Interface, d *appsv1.Deployment) error {
	return testutils.WaitForDeploymentComplete(c, d, e2elog.Logf, poll, pollLongTimeout)
}

// WaitForDeploymentCompleteAndCheckRolling waits for the deployment to complete, and check rolling update strategy isn't broken at any times.
// Rolling update strategy should not be broken during a rolling update.
func WaitForDeploymentCompleteAndCheckRolling(c clientset.Interface, d *appsv1.Deployment) error {
	return testutils.WaitForDeploymentCompleteAndCheckRolling(c, d, e2elog.Logf, poll, pollLongTimeout)
}

// WaitForDeploymentUpdatedReplicasGTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func WaitForDeploymentUpdatedReplicasGTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int32, desiredGeneration int64) error {
	return testutils.WaitForDeploymentUpdatedReplicasGTE(c, ns, deploymentName, minUpdatedReplicas, desiredGeneration, poll, pollLongTimeout)
}

// WaitForDeploymentRollbackCleared waits for given deployment either started rolling back or doesn't need to rollback.
// Note that rollback should be cleared shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRollbackCleared(c clientset.Interface, ns, deploymentName string) error {
	return testutils.WaitForDeploymentRollbackCleared(c, ns, deploymentName, poll, pollShortTimeout)
}

// WaitForDeploymentOldRSsNum waits for the deployment to clean up old rcs.
func WaitForDeploymentOldRSsNum(c clientset.Interface, ns, deploymentName string, desiredRSNum int) error {
	var oldRSs []*appsv1.ReplicaSet
	var d *appsv1.Deployment

	pollErr := wait.PollImmediate(poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		d = deployment

		_, oldRSs, err = deploymentutil.GetOldReplicaSets(deployment, c.AppsV1())
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

// WaitForDeploymentRevision waits for becoming the target revision of a delopyment.
func WaitForDeploymentRevision(c clientset.Interface, d *appsv1.Deployment, targetRevision string) error {
	err := wait.PollImmediate(poll, pollLongTimeout, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(d.Namespace).Get(d.Name, metav1.GetOptions{})
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
