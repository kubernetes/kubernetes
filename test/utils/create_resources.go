/*
Copyright 2018 The Kubernetes Authors.

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

// TODO: Refactor common part of functions in this file for generic object kinds.

package utils

import (
	"context"
	"fmt"
	"time"

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	storage "k8s.io/api/storage/v1"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
)

const (
	// Parameters for retrying with exponential backoff.
	retryBackoffInitialDuration = 100 * time.Millisecond
	retryBackoffFactor          = 3
	retryBackoffJitter          = 0
	retryBackoffSteps           = 6
)

// RetryWithExponentialBackOff is a utility for retrying the given function with exponential backoff.
func RetryWithExponentialBackOff(fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: retryBackoffInitialDuration,
		Factor:   retryBackoffFactor,
		Jitter:   retryBackoffJitter,
		Steps:    retryBackoffSteps,
	}
	return wait.ExponentialBackoff(backoff, fn)
}

// IsRetryableAPIError verifies if the error is liable for retry.
func IsRetryableAPIError(err error) bool {
	// These errors may indicate a transient error that we can retry in tests.
	if apierrors.IsInternalError(err) || apierrors.IsTimeout(err) || apierrors.IsServerTimeout(err) ||
		apierrors.IsTooManyRequests(err) || utilnet.IsProbableEOF(err) || utilnet.IsConnectionReset(err) {
		return true
	}
	// If the error sends the Retry-After header, we respect it as an explicit confirmation we should retry.
	if _, shouldRetry := apierrors.SuggestsClientDelay(err); shouldRetry {
		return true
	}
	return false
}

// CreatePodWithRetries creates a new Pod while retrying on failure.
func CreatePodWithRetries(c clientset.Interface, namespace string, obj *v1.Pod) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().Pods(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateRCWithRetries creates a new ReplicationController while retrying on failure.
func CreateRCWithRetries(c clientset.Interface, namespace string, obj *v1.ReplicationController) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().ReplicationControllers(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateReplicaSetWithRetries creates a new ReplicaSet retrying on failure.
func CreateReplicaSetWithRetries(c clientset.Interface, namespace string, obj *apps.ReplicaSet) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.AppsV1().ReplicaSets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateDeploymentWithRetries creates a new Deployment while retrying on failure.
func CreateDeploymentWithRetries(c clientset.Interface, namespace string, obj *apps.Deployment) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.AppsV1().Deployments(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateDaemonSetWithRetries creates a new DaemonSet while retrying on failure.
func CreateDaemonSetWithRetries(c clientset.Interface, namespace string, obj *apps.DaemonSet) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.AppsV1().DaemonSets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateJobWithRetries creates a new Job while retrying on failure.
func CreateJobWithRetries(c clientset.Interface, namespace string, obj *batch.Job) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.BatchV1().Jobs(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateSecretWithRetries creates a new Secret while retrying on failure.
func CreateSecretWithRetries(c clientset.Interface, namespace string, obj *v1.Secret) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().Secrets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateConfigMapWithRetries creates a new ConfigMap while retrying on failure.
func CreateConfigMapWithRetries(c clientset.Interface, namespace string, obj *v1.ConfigMap) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().ConfigMaps(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateServiceWithRetries creates a new Service while retrying on failure.
func CreateServiceWithRetries(c clientset.Interface, namespace string, obj *v1.Service) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().Services(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateStorageClassWithRetries creates a new StorageClass while retrying on failure.
func CreateStorageClassWithRetries(c clientset.Interface, obj *storage.StorageClass) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.StorageV1().StorageClasses().Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreateResourceQuotaWithRetries creates a new ResourceQuota while retrying on failure.
func CreateResourceQuotaWithRetries(c clientset.Interface, namespace string, obj *v1.ResourceQuota) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().ResourceQuotas(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreatePersistentVolumeWithRetries create a new PersistentVolume while retrying on failure.
func CreatePersistentVolumeWithRetries(c clientset.Interface, obj *v1.PersistentVolume) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().PersistentVolumes().Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}

// CreatePersistentVolumeClaimWithRetries creates a new PersistentVolumeClaim while retrying on failure.
func CreatePersistentVolumeClaimWithRetries(c clientset.Interface, namespace string, obj *v1.PersistentVolumeClaim) error {
	if obj == nil {
		return fmt.Errorf("Object provided to create is empty")
	}
	createFunc := func() (bool, error) {
		_, err := c.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to create object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(createFunc)
}
