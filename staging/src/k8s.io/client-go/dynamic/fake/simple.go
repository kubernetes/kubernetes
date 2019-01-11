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
	"strings"

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
	// In order to use List with this client, you have to have the v1.List registered in your scheme. Neat thing though
	// it does NOT have to be the *same* list
	scheme.AddKnownTypeWithName(schema.GroupVersionKind{Group: "fake-dynamic-client-group", Version: "v1", Kind: "List"}, &unstructured.UnstructuredList{})

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
	client    *FakeDynamicClient
	namespace string
	resource  schema.GroupVersionResource
}

var _ dynamic.Interface = &FakeDynamicClient{}

func (c *FakeDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource}
}

func (c *dynamicResourceClient) Namespace(ns string) dynamic.ResourceInterface {
	ret := *c
	ret.namespace = ns
	return &ret
}

func (c *dynamicResourceClient) Create(obj *unstructured.Unstructured, opts metav1.CreateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootCreateAction(c.resource, obj), obj)

	case len(c.namespace) == 0 && len(subresources) > 0:
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		name := accessor.GetName()
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootCreateSubresourceAction(c.resource, name, strings.Join(subresources, "/"), obj), obj)

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewCreateAction(c.resource, c.namespace, obj), obj)

	case len(c.namespace) > 0 && len(subresources) > 0:
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		name := accessor.GetName()
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewCreateSubresourceAction(c.resource, name, strings.Join(subresources, "/"), c.namespace, obj), obj)

	}

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

func (c *dynamicResourceClient) Update(obj *unstructured.Unstructured, opts metav1.UpdateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootUpdateAction(c.resource, obj), obj)

	case len(c.namespace) == 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootUpdateSubresourceAction(c.resource, strings.Join(subresources, "/"), obj), obj)

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewUpdateAction(c.resource, c.namespace, obj), obj)

	case len(c.namespace) > 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewUpdateSubresourceAction(c.resource, strings.Join(subresources, "/"), c.namespace, obj), obj)

	}

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

func (c *dynamicResourceClient) UpdateStatus(obj *unstructured.Unstructured, opts metav1.UpdateOptions) (*unstructured.Unstructured, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootUpdateSubresourceAction(c.resource, "status", obj), obj)

	case len(c.namespace) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewUpdateSubresourceAction(c.resource, "status", c.namespace, obj), obj)

	}

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

func (c *dynamicResourceClient) Delete(name string, opts *metav1.DeleteOptions, subresources ...string) error {
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		_, err = c.client.Fake.
			Invokes(testing.NewRootDeleteAction(c.resource, name), &metav1.Status{Status: "dynamic delete fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		_, err = c.client.Fake.
			Invokes(testing.NewRootDeleteSubresourceAction(c.resource, strings.Join(subresources, "/"), name), &metav1.Status{Status: "dynamic delete fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		_, err = c.client.Fake.
			Invokes(testing.NewDeleteAction(c.resource, c.namespace, name), &metav1.Status{Status: "dynamic delete fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		_, err = c.client.Fake.
			Invokes(testing.NewDeleteSubresourceAction(c.resource, strings.Join(subresources, "/"), c.namespace, name), &metav1.Status{Status: "dynamic delete fail"})
	}

	return err
}

func (c *dynamicResourceClient) DeleteCollection(opts *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	var err error
	switch {
	case len(c.namespace) == 0:
		action := testing.NewRootDeleteCollectionAction(c.resource, listOptions)
		_, err = c.client.Fake.Invokes(action, &metav1.Status{Status: "dynamic deletecollection fail"})

	case len(c.namespace) > 0:
		action := testing.NewDeleteCollectionAction(c.resource, c.namespace, listOptions)
		_, err = c.client.Fake.Invokes(action, &metav1.Status{Status: "dynamic deletecollection fail"})

	}

	return err
}

func (c *dynamicResourceClient) Get(name string, opts metav1.GetOptions, subresources ...string) (*unstructured.Unstructured, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootGetAction(c.resource, name), &metav1.Status{Status: "dynamic get fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootGetSubresourceAction(c.resource, strings.Join(subresources, "/"), name), &metav1.Status{Status: "dynamic get fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewGetAction(c.resource, c.namespace, name), &metav1.Status{Status: "dynamic get fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewGetSubresourceAction(c.resource, c.namespace, strings.Join(subresources, "/"), name), &metav1.Status{Status: "dynamic get fail"})
	}

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
	var obj runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0:
		obj, err = c.client.Fake.
			Invokes(testing.NewRootListAction(c.resource, schema.GroupVersionKind{Group: "fake-dynamic-client-group", Version: "v1", Kind: "" /*List is appended by the tracker automatically*/}, opts), &metav1.Status{Status: "dynamic list fail"})

	case len(c.namespace) > 0:
		obj, err = c.client.Fake.
			Invokes(testing.NewListAction(c.resource, schema.GroupVersionKind{Group: "fake-dynamic-client-group", Version: "v1", Kind: "" /*List is appended by the tracker automatically*/}, c.namespace, opts), &metav1.Status{Status: "dynamic list fail"})

	}

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
	list.SetResourceVersion(entireList.GetResourceVersion())
	for i := range entireList.Items {
		item := &entireList.Items[i]
		metadata, err := meta.Accessor(item)
		if err != nil {
			return nil, err
		}
		if label.Matches(labels.Set(metadata.GetLabels())) {
			list.Items = append(list.Items, *item)
		}
	}
	return list, nil
}

func (c *dynamicResourceClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	switch {
	case len(c.namespace) == 0:
		return c.client.Fake.
			InvokesWatch(testing.NewRootWatchAction(c.resource, opts))

	case len(c.namespace) > 0:
		return c.client.Fake.
			InvokesWatch(testing.NewWatchAction(c.resource, c.namespace, opts))

	}

	panic("math broke")
}

func (c *dynamicResourceClient) Patch(name string, pt types.PatchType, data []byte, opts metav1.UpdateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootPatchAction(c.resource, name, data), &metav1.Status{Status: "dynamic patch fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootPatchSubresourceAction(c.resource, name, data, subresources...), &metav1.Status{Status: "dynamic patch fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewPatchAction(c.resource, c.namespace, name, data), &metav1.Status{Status: "dynamic patch fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewPatchSubresourceAction(c.resource, c.namespace, name, data, subresources...), &metav1.Status{Status: "dynamic patch fail"})

	}

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
