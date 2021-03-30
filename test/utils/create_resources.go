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

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	storage "k8s.io/api/storage/v1"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

func CreatePod(c clientset.Interface, namespace string, obj *v1.Pod) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().Pods(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object: %v", err)
}

func CreateRC(c clientset.Interface, namespace string, obj *v1.ReplicationController) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().ReplicationControllers(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object: %v", err)
}

func CreateReplicaSet(c clientset.Interface, namespace string, obj *apps.ReplicaSet) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.AppsV1().ReplicaSets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object: %v", err)
}

func CreateDeployment(c clientset.Interface, namespace string, obj *apps.Deployment) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.AppsV1().Deployments(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object: %v", err)
}

func CreateDaemonSet(c clientset.Interface, namespace string, obj *apps.DaemonSet) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.AppsV1().DaemonSets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object: %v", err)
}

func CreateJob(c clientset.Interface, namespace string, obj *batch.Job) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.BatchV1().Jobs(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreateSecret(c clientset.Interface, namespace string, obj *v1.Secret) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().Secrets(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreateConfigMap(c clientset.Interface, namespace string, obj *v1.ConfigMap) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().ConfigMaps(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreateService(c clientset.Interface, namespace string, obj *v1.Service) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().Services(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreateStorageClass(c clientset.Interface, obj *storage.StorageClass) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.StorageV1().StorageClasses().Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreateResourceQuota(c clientset.Interface, namespace string, obj *v1.ResourceQuota) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().ResourceQuotas(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreatePersistentVolume(c clientset.Interface, obj *v1.PersistentVolume) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().PersistentVolumes().Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}

func CreatePersistentVolumeClaim(c clientset.Interface, namespace string, obj *v1.PersistentVolumeClaim) error {
	if obj == nil {
		return fmt.Errorf("object provided to create is empty")
	}
	_, err := c.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(), obj, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("failed to create object with non-retriable error: %v", err)
}
