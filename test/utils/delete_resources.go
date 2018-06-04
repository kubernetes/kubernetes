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
	"fmt"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
)

func deleteResource(c clientset.Interface, kind schema.GroupKind, namespace, name string, options *metav1.DeleteOptions) error {
	switch kind {
	case api.Kind("Pod"):
		return c.CoreV1().Pods(namespace).Delete(name, options)
	case api.Kind("ReplicationController"):
		return c.CoreV1().ReplicationControllers(namespace).Delete(name, options)
	case extensionsinternal.Kind("ReplicaSet"), appsinternal.Kind("ReplicaSet"):
		return c.ExtensionsV1beta1().ReplicaSets(namespace).Delete(name, options)
	case extensionsinternal.Kind("Deployment"), appsinternal.Kind("Deployment"):
		return c.ExtensionsV1beta1().Deployments(namespace).Delete(name, options)
	case extensionsinternal.Kind("DaemonSet"):
		return c.ExtensionsV1beta1().DaemonSets(namespace).Delete(name, options)
	case batchinternal.Kind("Job"):
		return c.BatchV1().Jobs(namespace).Delete(name, options)
	case api.Kind("Secret"):
		return c.CoreV1().Secrets(namespace).Delete(name, options)
	case api.Kind("ConfigMap"):
		return c.CoreV1().ConfigMaps(namespace).Delete(name, options)
	case api.Kind("Service"):
		return c.CoreV1().Services(namespace).Delete(name, options)
	default:
		return fmt.Errorf("Unsupported kind when deleting: %v", kind)
	}
}

func DeleteResourceWithRetries(c clientset.Interface, kind schema.GroupKind, namespace, name string, options *metav1.DeleteOptions) error {
	deleteFunc := func() (bool, error) {
		err := deleteResource(c, kind, namespace, name, options)
		if err == nil || apierrs.IsNotFound(err) {
			return true, nil
		}
		if IsRetryableAPIError(err) {
			return false, nil
		}
		return false, fmt.Errorf("Failed to delete object with non-retriable error: %v", err)
	}
	return RetryWithExponentialBackOff(deleteFunc)
}
