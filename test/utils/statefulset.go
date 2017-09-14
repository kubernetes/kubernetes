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

	apps "k8s.io/api/apps/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
)

// TODO(juntee): add LogPodsOfStatefulSet function

// Waits for the statefulset status to become valid.
// Note that the status should stay valid at all times unless shortly after a scaling event or the statefulset is just created.
func WaitForStatefulSetStatusValid(c clientset.Interface, s *apps.StatefulSet, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var (
		statefulset *apps.StatefulSet
		reason      string
	)

	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		statefulset, err = c.Apps().StatefulSets(s.Namespace).Get(s.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// When the statefulset status and its underlying resources reach the desired state, we're done
		// TODO(juntee): confirm desired state's status
		newStatus := statefulset.Status
		if newStatus.UpdatedReplicas == *(statefulset.Spec.Replicas) &&
			newStatus.Replicas == *(statefulset.Spec.Replicas) &&
			newStatus.ObservedGeneration >= statefulset.Generation {
			return true, nil
		}

		reason = fmt.Sprintf("statefulset status: %#v", statefulset.Status)
		logf(reason)

		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("%s", reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for statefulset %q status to match expectation: %v", s.Name, err)
	}
	return nil
}

// WaitForStatefulSetImage waits for the statefulset's container image to match the given image.
// We wait for 1 minute here to fail early.
func WaitForStatefulSetImage(c clientset.Interface, ns, statefulsetName string, revision, image string, logf LogfFn, pollInterval, pollTimeout time.Duration) error {
	var statefulset *apps.StatefulSet
	var reason string
	err := wait.Poll(pollInterval, pollTimeout, func() (bool, error) {
		var err error
		statefulset, err = c.Apps().StatefulSets(ns).Get(statefulsetName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if statefulset.Spec.Template.Spec.Containers[0].Image != image {
			reason = fmt.Sprintf("StatefulSet %q doesn't have the required image set", statefulset.Name)
			logf(reason)
			return false, nil
		}
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf(reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for statefulset %q (got %s) image to match expectation (expected %s): %v", statefulsetName, statefulset.Spec.Template.Spec.Containers[0].Image, image, err)
	}
	return nil
}
