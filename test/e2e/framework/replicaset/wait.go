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
	"fmt"

	"github.com/onsi/ginkgo"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	appsclient "k8s.io/client-go/kubernetes/typed/apps/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

// RunReplicaSet launches (and verifies correctness) of a replicaset.
func RunReplicaSet(config testutils.ReplicaSetConfig) error {
	ginkgo.By(fmt.Sprintf("creating replicaset %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = framework.DumpNodeDebugInfo
	config.ContainerDumpFunc = framework.LogFailedContainers
	return testutils.RunReplicaSet(config)
}

// WaitForReadyReplicaSet waits until the replicaset has all of its replicas ready.
func WaitForReadyReplicaSet(c clientset.Interface, ns, name string) error {
	err := wait.Poll(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return *(rs.Spec.Replicas) == rs.Status.Replicas && *(rs.Spec.Replicas) == rs.Status.ReadyReplicas, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never became ready", name)
	}
	return err
}

// WaitForReplicaSetDesiredReplicas waits until the replicaset has desired number of replicas.
func WaitForReplicaSetDesiredReplicas(rsClient appsclient.ReplicaSetsGetter, replicaSet *appsv1.ReplicaSet) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := rsClient.ReplicaSets(replicaSet.Namespace).Get(replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.Replicas == *(replicaSet.Spec.Replicas) && rs.Status.Replicas == *(rs.Spec.Replicas), nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never had desired number of replicas", replicaSet.Name)
	}
	return err
}

// WaitForReplicaSetTargetSpecReplicas waits for .spec.replicas of a RS to equal targetReplicaNum
func WaitForReplicaSetTargetSpecReplicas(c clientset.Interface, replicaSet *appsv1.ReplicaSet, targetReplicaNum int32) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(replicaSet.Namespace).Get(replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && *rs.Spec.Replicas == targetReplicaNum, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never had desired number of .spec.replicas", replicaSet.Name)
	}
	return err
}

// WaitForReplicaSetTargetAvailableReplicas waits for .status.availableReplicas of a RS to equal targetReplicaNum
func WaitForReplicaSetTargetAvailableReplicas(c clientset.Interface, replicaSet *appsv1.ReplicaSet, targetReplicaNum int32) error {
	desiredGeneration := replicaSet.Generation
	err := wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
		rs, err := c.AppsV1().ReplicaSets(replicaSet.Namespace).Get(replicaSet.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration && rs.Status.AvailableReplicas == targetReplicaNum, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replicaset %q never had desired number of .status.availableReplicas", replicaSet.Name)
	}
	return err
}
