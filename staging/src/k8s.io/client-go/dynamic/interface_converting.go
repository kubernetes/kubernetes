/*
Copyright 2021 The Kubernetes Authors.

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

package dynamic

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
)

type ConvertingInterface interface {
	Resource(resource schema.GroupVersionResource) ConvertingNamespaceableResourceInterface
}

type ConvertingResourceInterface interface {
	Create(ctx context.Context, obj runtime.Object, options metav1.CreateOptions, subresources ...string) (runtime.Object, error)
	Update(ctx context.Context, obj runtime.Object, options metav1.UpdateOptions, subresources ...string) (runtime.Object, error)
	UpdateStatus(ctx context.Context, obj runtime.Object, options metav1.UpdateOptions) (runtime.Object, error)
	Delete(ctx context.Context, name string, options metav1.DeleteOptions, subresources ...string) error
	DeleteCollection(ctx context.Context, options metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(ctx context.Context, obj runtime.Object, name string, options metav1.GetOptions, subresources ...string) (runtime.Object, error)
	List(ctx context.Context, obj runtime.Object, opts metav1.ListOptions) (runtime.Object, error)
	Watch(ctx context.Context, obj runtime.Object, opts metav1.ListOptions) (watch.Interface, error)
	Patch(ctx context.Context, obj runtime.Object, name string, pt types.PatchType, data []byte, options metav1.PatchOptions, subresources ...string) (runtime.Object, error)
}

type ConvertingNamespaceableResourceInterface interface {
	Namespace(string) ConvertingResourceInterface
	ConvertingResourceInterface
}
