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

	extensions "k8s.io/api/extensions/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	testutils "k8s.io/kubernetes/test/utils"
)

type updateRsFunc func(d *extensions.ReplicaSet)

func UpdateReplicaSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate testutils.UpdateReplicaSetFunc) (*extensions.ReplicaSet, error) {
	return testutils.UpdateReplicaSetWithRetries(c, namespace, name, applyUpdate, Logf, Poll, pollShortTimeout)
}

// CheckNewRSAnnotations check if the new RS's annotation is as expected
func CheckNewRSAnnotations(c clientset.Interface, ns, deploymentName string, expectedAnnotations map[string]string) error {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c.ExtensionsV1beta1())
	if err != nil {
		return err
	}
	for k, v := range expectedAnnotations {
		// Skip checking revision annotations
		if k != deploymentutil.RevisionAnnotation && v != newRS.Annotations[k] {
			return fmt.Errorf("Expected new RS annotations = %+v, got %+v", expectedAnnotations, newRS.Annotations)
		}
	}
	return nil
}

// Delete a ReplicaSet and all pods it spawned
func DeleteReplicaSet(clientset clientset.Interface, internalClientset internalclientset.Interface, ns, name string) error {
	By(fmt.Sprintf("deleting ReplicaSet %s in namespace %s", name, ns))
	rs, err := clientset.Extensions().ReplicaSets(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("ReplicaSet %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	startTime := time.Now()
	err = clientset.ExtensionsV1beta1().ReplicaSets(ns).Delete(name, &metav1.DeleteOptions{})
	if apierrs.IsNotFound(err) {
		Logf("ReplicaSet %s was already deleted: %v", name, err)
		return nil
	}
	deleteRSTime := time.Now().Sub(startTime)
	Logf("Deleting RS %s took: %v", name, deleteRSTime)
	if err == nil {
		err = waitForReplicaSetPodsGone(clientset, rs)
	}
	terminatePodTime := time.Now().Sub(startTime) - deleteRSTime
	Logf("Terminating ReplicaSet %s pods took: %v", name, terminatePodTime)
	return err
}

// waitForReplicaSetPodsGone waits until there are no pods reported under a
// ReplicaSet selector (because the pods have completed termination).
func waitForReplicaSetPodsGone(c clientset.Interface, rs *extensions.ReplicaSet) error {
	return wait.PollImmediate(Poll, 2*time.Minute, func() (bool, error) {
		selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
		ExpectNoError(err)
		options := metav1.ListOptions{LabelSelector: selector.String()}
		if pods, err := c.Core().Pods(rs.Namespace).List(options); err == nil && len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
}

// WaitForReadyReplicaSet waits until the replica set has all of its replicas ready.
func WaitForReadyReplicaSet(c clientset.Interface, ns, name string) error {
	err := wait.Poll(Poll, pollShortTimeout, func() (bool, error) {
		rs, err := c.Extensions().ReplicaSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return *(rs.Spec.Replicas) == rs.Status.Replicas && *(rs.Spec.Replicas) == rs.Status.ReadyReplicas, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("replica set %q never became ready", name)
	}
	return err
}

func RunReplicaSet(config testutils.ReplicaSetConfig) error {
	By(fmt.Sprintf("creating replicaset %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = DumpNodeDebugInfo
	config.ContainerDumpFunc = LogFailedContainers
	return testutils.RunReplicaSet(config)
}
