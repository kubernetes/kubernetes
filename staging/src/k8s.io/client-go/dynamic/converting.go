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
	"reflect"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
)

type convertingDynamicClient struct {
	dynamicClient Interface
}

// NewForConfig creates a new convertingDynamic client or returns an error.
func NewConvertingDynamicClientForConfig(inConfig *rest.Config) (ConvertingInterface, error) {
	dynamicClient, err := NewForConfig(inConfig)
	if err != nil {
		return nil, err
	}

	return &convertingDynamicClient{dynamicClient: dynamicClient}, nil
}

type convertingDynamicResourceClient struct {
	client    ResourceInterface
	namespace string
	resource  schema.GroupVersionResource
}

func (c *convertingDynamicClient) Resource(resource schema.GroupVersionResource) ConvertingNamespaceableResourceInterface {
	return &convertingDynamicResourceClient{client: c.dynamicClient.Resource(resource), resource: resource}
}

func (c *convertingDynamicResourceClient) Namespace(ns string) ConvertingResourceInterface {
	ret := *c
	ret.namespace = ns
	return &ret
}

func unstructuredFromObj(obj runtime.Object) (*unstructured.Unstructured, error) {
	objectMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	unstructuredIn := &unstructured.Unstructured{
		Object: objectMap,
	}
	return unstructuredIn, nil
}

func unstructuredToRuntimeObject(unstructuredObj *unstructured.Unstructured, targetType runtime.Object) (runtime.Object, error) {
	castObj := reflect.New(reflect.TypeOf(targetType).Elem()).Interface().(runtime.Object).(runtime.Object)
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredObj.Object, castObj); err != nil {
		return nil, err
	}
	return castObj, nil
}

func unstructuredListToRuntimeObject(unstructuredObj *unstructured.UnstructuredList, targetType runtime.Object) (runtime.Object, error) {
	castObj := reflect.New(reflect.TypeOf(targetType).Elem()).Interface().(runtime.Object).(runtime.Object)
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredObj.Object, castObj); err != nil {
		return nil, err
	}
	return castObj, nil
}

func (c *convertingDynamicResourceClient) Create(ctx context.Context, obj runtime.Object, opts metav1.CreateOptions, subresources ...string) (runtime.Object, error) {
	unstructuredIn, err := unstructuredFromObj(obj)
	if err != nil {
		return nil, err
	}
	unstructuredOut, err := c.client.Create(ctx, unstructuredIn, opts, subresources...)
	if err != nil {
		return nil, err
	}

	return unstructuredToRuntimeObject(unstructuredOut, obj)
}

func (c *convertingDynamicResourceClient) Update(ctx context.Context, obj runtime.Object, opts metav1.UpdateOptions, subresources ...string) (runtime.Object, error) {
	unstructuredIn, err := unstructuredFromObj(obj)
	if err != nil {
		return nil, err
	}
	unstructuredOut, err := c.client.Update(ctx, unstructuredIn, opts, subresources...)
	if err != nil {
		return nil, err
	}

	return unstructuredToRuntimeObject(unstructuredOut, obj)
}

func (c *convertingDynamicResourceClient) UpdateStatus(ctx context.Context, obj runtime.Object, opts metav1.UpdateOptions) (runtime.Object, error) {
	unstructuredIn, err := unstructuredFromObj(obj)
	if err != nil {
		return nil, err
	}
	unstructuredOut, err := c.client.UpdateStatus(ctx, unstructuredIn, opts)
	if err != nil {
		return nil, err
	}

	return unstructuredToRuntimeObject(unstructuredOut, obj)
}

func (c *convertingDynamicResourceClient) Delete(ctx context.Context, name string, opts metav1.DeleteOptions, subresources ...string) error {
	return c.client.Delete(ctx, name, opts, subresources...)
}

func (c *convertingDynamicResourceClient) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	return c.client.DeleteCollection(ctx, opts, listOptions)
}

func (c *convertingDynamicResourceClient) Get(ctx context.Context, obj runtime.Object, name string, opts metav1.GetOptions, subresources ...string) (runtime.Object, error) {
	unstructuredOut, err := c.client.Get(ctx, name, opts, subresources...)
	if err != nil {
		return nil, err
	}
	return unstructuredToRuntimeObject(unstructuredOut, obj)
}

func (c *convertingDynamicResourceClient) List(ctx context.Context, obj runtime.Object, opts metav1.ListOptions) (runtime.Object, error) {
	unstructuredOutList, err := c.client.List(ctx, opts)
	if err != nil {
		return nil, err
	}
	return unstructuredListToRuntimeObject(unstructuredOutList, obj)
}

func (c *convertingDynamicResourceClient) Watch(ctx context.Context, obj runtime.Object, opts metav1.ListOptions) (watch.Interface, error) {
	unstructuredWatch, err := c.client.Watch(ctx, opts)
	if err != nil {
		return nil, err
	}

	return watch.Filter(unstructuredWatch, castingFilter{obj: obj}.castToObj), nil
}

type castingFilter struct {
	obj runtime.Object
}

func (c castingFilter) castToObj(in watch.Event) (watch.Event, bool) {
	keep := true
	if in.Object == nil {
		return in, keep
	}

	out := watch.Event{
		Type: in.Type,
	}
	castObj, err := unstructuredToRuntimeObject(out.Object.(*unstructured.Unstructured), c.obj)
	if err != nil {
		utilruntime.HandleError(err)
		// TODO create error in watch stream.  This will cause a cast error
		return in, true
	}

	out.Object = castObj
	return out, true
}

func (c *convertingDynamicResourceClient) Patch(ctx context.Context, obj runtime.Object, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (runtime.Object, error) {
	unstructuredOut, err := c.client.Patch(ctx, name, pt, data, opts, subresources...)
	if err != nil {
		return nil, err
	}

	return unstructuredToRuntimeObject(unstructuredOut, obj)
}
