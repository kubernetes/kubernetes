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

package fake

import (
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/testing"
)

func NewSimpleDynamicClient(scheme *runtime.Scheme, objects ...runtime.Object) *FakeDynamicClient {
	codecs := serializer.NewCodecFactory(scheme)
	o := testing.NewObjectTracker(scheme, codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	cs := &FakeDynamicClient{}
	cs.AddReactor("*", "*", testing.ObjectReaction(o))
	cs.AddWatchReactor("*", func(action testing.Action) (handled bool, ret watch.Interface, err error) {
		gvr := action.GetResource()
		ns := action.GetNamespace()
		watch, err := o.Watch(gvr, ns)
		if err != nil {
			return false, nil, err
		}
		return true, watch, nil
	})

	return cs
}

// Clientset implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type FakeDynamicClient struct {
	testing.Fake
	scheme *runtime.Scheme
}

type dynamicResourceClient struct {
	client      *FakeDynamicClient
	namespace   string
	resource    schema.GroupVersionResource
	subresource string
}

var _ dynamic.DynamicInterface = &FakeDynamicClient{}

func (c *FakeDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableDynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource}
}

// Deprecated, this isn't how we want to do it
func (c *FakeDynamicClient) ClusterSubresource(resource schema.GroupVersionResource, subresource string) dynamic.DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource, subresource: subresource}
}

// Deprecated, this isn't how we want to do it
func (c *FakeDynamicClient) NamespacedSubresource(resource schema.GroupVersionResource, subresource, namespace string) dynamic.DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource, namespace: namespace, subresource: subresource}
}

func (c *dynamicResourceClient) Namespace(ns string) dynamic.DynamicResourceInterface {
	ret := *c
	ret.namespace = ns
	return &ret
}

func (c *dynamicResourceClient) Create(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	uncastRet, err := c.client.Fake.
		Invokes(testing.NewCreateAction(c.resource, c.namespace, obj), obj)

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}

	ret := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(uncastRet, ret, nil); err != nil {
		return nil, err
	}
	return ret, err
}

func (c *dynamicResourceClient) Update(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	uncastRet, err := c.client.Fake.
		Invokes(testing.NewUpdateAction(c.resource, c.namespace, obj), obj)

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}

	ret := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(uncastRet, ret, nil); err != nil {
		return nil, err
	}
	return ret, err
}

func (c *dynamicResourceClient) UpdateStatus(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	uncastRet, err := c.client.Fake.
		Invokes(testing.NewUpdateSubresourceAction(c.resource, "status", c.namespace, obj), obj)

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}

	ret := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(uncastRet, ret, nil); err != nil {
		return nil, err
	}
	return ret, err
}

func (c *dynamicResourceClient) Delete(name string, opts *metav1.DeleteOptions) error {
	_, err := c.client.Fake.
		Invokes(testing.NewDeleteAction(c.resource, c.namespace, name), &metav1.Status{Status: "dynamic delete fail"})

	return err
}

func (c *dynamicResourceClient) DeleteCollection(opts *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(c.resource, c.namespace, listOptions)

	_, err := c.client.Fake.Invokes(action, &metav1.Status{Status: "dynamic deletecollection fail"})
	return err
}

func (c *dynamicResourceClient) Get(name string, opts metav1.GetOptions) (*unstructured.Unstructured, error) {
	uncastRet, err := c.client.Fake.
		Invokes(testing.NewGetAction(c.resource, c.namespace, name), &metav1.Status{Status: "dynamic get fail"})

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}

	ret := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(uncastRet, ret, nil); err != nil {
		return nil, err
	}
	return ret, err
}

func (c *dynamicResourceClient) List(opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	obj, err := c.client.Fake.
		Invokes(testing.NewListAction(c.resource, schema.GroupVersionKind{Version: "v1", Kind: "List"}, c.namespace, opts), &metav1.Status{Status: "dynamic list fail"})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}

	retUnstructured := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(obj, retUnstructured, nil); err != nil {
		return nil, err
	}
	entireList, err := retUnstructured.ToList()
	if err != nil {
		return nil, err
	}

	list := &unstructured.UnstructuredList{}
	for _, item := range entireList.Items {
		metadata, err := meta.Accessor(item)
		if err != nil {
			return nil, err
		}
		if label.Matches(labels.Set(metadata.GetLabels())) {
			list.Items = append(list.Items, item)
		}
	}
	return list, nil
}

func (c *dynamicResourceClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	return c.client.Fake.
		InvokesWatch(testing.NewWatchAction(c.resource, c.namespace, opts))
}

func (c *dynamicResourceClient) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (*unstructured.Unstructured, error) {
	uncastRet, err := c.client.Fake.
		Invokes(testing.NewPatchSubresourceAction(c.resource, c.namespace, name, data, subresources...), &metav1.Status{Status: "dynamic patch fail"})

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}

	ret := &unstructured.Unstructured{}
	if err := c.client.scheme.Convert(uncastRet, ret, nil); err != nil {
		return nil, err
	}
	return ret, err
}
