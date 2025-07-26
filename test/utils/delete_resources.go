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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
)

// ResourceDeleter interface for common delete operations
type ResourceDeleter interface {
	Delete(context.Context, string, metav1.DeleteOptions) error
}

// deleteGenericResource handles deletion for any resource implementing ResourceDeleter
func deleteGenericResource(ctx context.Context, deleter ResourceDeleter, name string, options metav1.DeleteOptions) error {
	return deleter.Delete(ctx, name, options)
}

func DeleteResource(c clientset.Interface, kind schema.GroupKind, namespace, name string, options metav1.DeleteOptions) error {
	ctx := context.TODO()
	switch kind {
	case api.Kind("Pod"):
		return deleteGenericResource(ctx, c.CoreV1().Pods(namespace), name, options)
	case api.Kind("ReplicationController"):
		return deleteGenericResource(ctx, c.CoreV1().ReplicationControllers(namespace), name, options)
	case extensionsinternal.Kind("ReplicaSet"), appsinternal.Kind("ReplicaSet"):
		return deleteGenericResource(ctx, c.AppsV1().ReplicaSets(namespace), name, options)
	case extensionsinternal.Kind("Deployment"), appsinternal.Kind("Deployment"):
		return deleteGenericResource(ctx, c.AppsV1().Deployments(namespace), name, options)
	case extensionsinternal.Kind("DaemonSet"):
		return deleteGenericResource(ctx, c.AppsV1().DaemonSets(namespace), name, options)
	case batchinternal.Kind("Job"):
		return deleteGenericResource(ctx, c.BatchV1().Jobs(namespace), name, options)
	case api.Kind("Secret"):
		return deleteGenericResource(ctx, c.CoreV1().Secrets(namespace), name, options)
	case api.Kind("ConfigMap"):
		return deleteGenericResource(ctx, c.CoreV1().ConfigMaps(namespace), name, options)
	case api.Kind("Service"):
		return deleteGenericResource(ctx, c.CoreV1().Services(namespace), name, options)
	default:
		return fmt.Errorf("unsupported kind when deleting: %v", kind)
	}
}
