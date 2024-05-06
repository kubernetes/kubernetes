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

package replicaset

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

// WaitForReadyReplicaSet waits until the replicaset has all of its replicas ready.
func WaitForReadyReplicaSet(ctx context.Context, c clientset.Interface, ns, name string) error {
	err := wait.PollWithContext(ctx, framework.Poll, framework.PodStartTimeout, func(ctx context.Context) (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return *(rs.Spec.Replicas) == rs.Status.Replicas && *(rs.Spec.Replicas) == rs.Status.ReadyReplicas, nil
	})
	if wait.Interrupted(err) {
		err = fmt.Errorf("replicaset %q never became ready", name)
	}
	return err
}

// WaitForReplicaSetTargetAvailableReplicas waits for .status.availableReplicas of a RS to equal targetReplicaNum
func WaitForReplicaSetTargetAvailableReplicas(ctx context.Context, c clientset.Interface, replicaSet *appsv1.ReplicaSet, targetReplicaNum int32) error {
	return WaitForReplicaSetTargetAvailableReplicasWithTimeout(ctx, c, replicaSet, targetReplicaNum, framework.PodStartTimeout)
}

// WaitForReplicaSetTargetAvailableReplicasWithTimeout waits for .status.availableReplicas of a RS to equal targetReplicaNum
// with given timeout.
func WaitForReplicaSetTargetAvailableReplicasWithTimeout(ctx context.Context, c clientset.Interface, replicaSet *appsv1.ReplicaSet, targetReplicaNum int32, timeout time.Duration) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(replicaSet.Namespace).Get(ctx, replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.AvailableReplicas == targetReplicaNum, nil
	})
	if wait.Interrupted(err) {
		err = fmt.Errorf("replicaset %q never had desired number of .status.availableReplicas", replicaSet.Name)
	}
	return err
}
