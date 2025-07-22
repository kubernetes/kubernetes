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

package resource

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
)

var (
	kindReplicationController = schema.GroupKind{Kind: "ReplicationController"}
	kindExtensionsReplicaSet  = schema.GroupKind{Group: "extensions", Kind: "ReplicaSet"}
	kindAppsReplicaSet        = schema.GroupKind{Group: "apps", Kind: "ReplicaSet"}
	kindExtensionsDeployment  = schema.GroupKind{Group: "extensions", Kind: "Deployment"}
	kindAppsDeployment        = schema.GroupKind{Group: "apps", Kind: "Deployment"}
	kindExtensionsDaemonSet   = schema.GroupKind{Group: "extensions", Kind: "DaemonSet"}
	kindBatchJob              = schema.GroupKind{Group: "batch", Kind: "Job"}
)

// GetRuntimeObjectForKind returns a runtime.Object based on its GroupKind,
// namespace and name.
func GetRuntimeObjectForKind(ctx context.Context, c clientset.Interface, kind schema.GroupKind, ns, name string) (runtime.Object, error) {
	switch kind {
	case kindReplicationController:
		return c.CoreV1().ReplicationControllers(ns).Get(ctx, name, metav1.GetOptions{})
	case kindExtensionsReplicaSet, kindAppsReplicaSet:
		return c.AppsV1().ReplicaSets(ns).Get(ctx, name, metav1.GetOptions{})
	case kindExtensionsDeployment, kindAppsDeployment:
		return c.AppsV1().Deployments(ns).Get(ctx, name, metav1.GetOptions{})
	case kindExtensionsDaemonSet:
		return c.AppsV1().DaemonSets(ns).Get(ctx, name, metav1.GetOptions{})
	case kindBatchJob:
		return c.BatchV1().Jobs(ns).Get(ctx, name, metav1.GetOptions{})
	default:
		return nil, fmt.Errorf("Unsupported kind when getting runtime object: %v", kind)
	}
}

// GetSelectorFromRuntimeObject returns the labels for the given object.
func GetSelectorFromRuntimeObject(obj runtime.Object) (labels.Selector, error) {
	switch typed := obj.(type) {
	case *v1.ReplicationController:
		return labels.SelectorFromSet(typed.Spec.Selector), nil
	case *extensionsv1beta1.ReplicaSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *appsv1.ReplicaSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *extensionsv1beta1.Deployment:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *appsv1.Deployment:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *extensionsv1beta1.DaemonSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *appsv1.DaemonSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *batchv1.Job:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *autoscalingv1.Scale:
		selector, err := metav1.ParseToLabelSelector(typed.Status.Selector)
		if err != nil {
			return nil, fmt.Errorf("Parsing selector for: %v encountered an error: %w", obj, err)
		}
		return metav1.LabelSelectorAsSelector(selector)
	default:
		return nil, fmt.Errorf("Unsupported kind when getting selector: %v", obj)
	}
}

// GetReplicasFromRuntimeObject returns the number of replicas for the given
// object.
func GetReplicasFromRuntimeObject(obj runtime.Object) (int32, error) {
	switch typed := obj.(type) {
	case *v1.ReplicationController:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensionsv1beta1.ReplicaSet:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *appsv1.ReplicaSet:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensionsv1beta1.Deployment:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *appsv1.Deployment:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensionsv1beta1.DaemonSet:
		return 0, nil
	case *appsv1.DaemonSet:
		return 0, nil
	case *batchv1.Job:
		// TODO: currently we use pause pods so that's OK. When we'll want to switch to Pods
		// that actually finish we need a better way to do this.
		if typed.Spec.Parallelism != nil {
			return *typed.Spec.Parallelism, nil
		}
		return 0, nil
	case *autoscalingv1.Scale:
		return typed.Status.Replicas, nil
	default:
		return -1, fmt.Errorf("Unsupported kind when getting number of replicas: %v", obj)
	}
}
