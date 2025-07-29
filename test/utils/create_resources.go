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

package utils

import (
	"context"
	"fmt"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

// Utility for retrying the given function with exponential backoff.
func RetryWithExponentialBackOff(fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: retryBackoffInitialDuration,
		Factor:   retryBackoffFactor,
		Jitter:   retryBackoffJitter,
		Steps:    retryBackoffSteps,
	}
	return wait.ExponentialBackoff(backoff, fn)
}

// createWithRetries is a generic function to create Kubernetes resources with retries.
func createWithRetries[T metav1.Object](
	ctx context.Context,
	create func() (T, error),
) error {
	if create == nil {
		return fmt.Errorf("create function provided to create is nil")
	}
	return RetryWithExponentialBackOff(func() (bool, error) {
		obj, err := create()
		if isGenerateNameConflict(obj, err) {
			return false, nil
		}
		if err == nil || apierrors.IsAlreadyExists(err) {
			return true, nil
		}
		return false, fmt.Errorf("create %T failed: %v", obj, err)
	})
}

func CreatePodWithRetries(ctx context.Context, c clientset.Interface, ns string, pod *v1.Pod) error {
	return createWithRetries(ctx, func() (*v1.Pod, error) {
		return c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
	})
}

func CreateRCWithRetries(ctx context.Context, c clientset.Interface, namespace string, obj *v1.ReplicationController) error {
	return createWithRetries(ctx, func() (*v1.ReplicationController, error) {
		return c.CoreV1().ReplicationControllers(namespace).Create(ctx, obj, metav1.CreateOptions{})
	})
}

func CreateReplicaSetWithRetries(ctx context.Context, c clientset.Interface, namespace string, obj *apps.ReplicaSet) error {
	return createWithRetries(ctx, func() (*apps.ReplicaSet, error) {
		return c.AppsV1().ReplicaSets(namespace).Create(ctx, obj, metav1.CreateOptions{})
	})
}

func CreateDeploymentWithRetries(ctx context.Context, c clientset.Interface, namespace string, obj *apps.Deployment) error {
	return createWithRetries(ctx, func() (*apps.Deployment, error) {
		return c.AppsV1().Deployments(namespace).Create(ctx, obj, metav1.CreateOptions{})
	})
}

func CreateServiceWithRetries(ctx context.Context, c clientset.Interface, namespace string, obj *v1.Service) error {
	return createWithRetries(ctx, func() (*v1.Service, error) {
		return c.CoreV1().Services(namespace).Create(ctx, obj, metav1.CreateOptions{})
	})
}

func CreatePersistentVolumeWithRetries(ctx context.Context, c clientset.Interface, obj *v1.PersistentVolume) error {
	return createWithRetries(ctx, func() (*v1.PersistentVolume, error) {
		return c.CoreV1().PersistentVolumes().Create(ctx, obj, metav1.CreateOptions{})
	})
}

func CreatePersistentVolumeClaimWithRetries(ctx context.Context, c clientset.Interface, namespace string, obj *v1.PersistentVolumeClaim) error {
	return createWithRetries(ctx, func() (*v1.PersistentVolumeClaim, error) {
		return c.CoreV1().PersistentVolumeClaims(namespace).Create(ctx, obj, metav1.CreateOptions{})
	})
}

// isGenerateNameConflict returns whether the error is generateName conflict or not.
func isGenerateNameConflict(objMeta metav1.Object, err error) bool {
	// Only retry if this was a generateName collision (i.e. Name was empty)
	return apierrors.IsAlreadyExists(err) && objMeta.GetName() == ""
}
