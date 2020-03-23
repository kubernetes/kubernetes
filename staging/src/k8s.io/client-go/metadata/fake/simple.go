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
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/testing"
)

// MetadataClient assists in creating fake objects for use when testing, since metadata.Getter
// does not expose create
type MetadataClient interface {
	metadata.Getter
	CreateFake(obj *metav1.PartialObjectMetadata, opts metav1.CreateOptions, subresources ...string) (*metav1.PartialObjectMetadata, error)
	UpdateFake(obj *metav1.PartialObjectMetadata, opts metav1.UpdateOptions, subresources ...string) (*metav1.PartialObjectMetadata, error)
}

// NewSimpleMetadataClient creates a new client that will use the provided scheme and respond with the
// provided objects when requests are made. It will track actions made to the client which can be checked
// with GetActions().
func NewSimpleMetadataClient(scheme *runtime.Scheme, objects ...runtime.Object) *FakeMetadataClient {
	gvkFakeList := schema.GroupVersionKind{Group: "fake-metadata-client-group", Version: "v1", Kind: "List"}
	if !scheme.Recognizes(gvkFakeList) {
		// In order to use List with this client, you have to have the v1.List registered in your scheme, since this is a test
		// type we modify the input scheme
		scheme.AddKnownTypeWithName(gvkFakeList, &metav1.List{})
	}

	codecs := serializer.NewCodecFactory(scheme)
	o := testing.NewObjectTracker(scheme, codecs.UniversalDeserializer())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	cs := &FakeMetadataClient{scheme: scheme}
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

// FakeMetadataClient implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type FakeMetadataClient struct {
	testing.Fake
	scheme *runtime.Scheme
}

type metadataResourceClient struct {
	client    *FakeMetadataClient
	namespace string
	resource  schema.GroupVersionResource
}

var _ metadata.Interface = &FakeMetadataClient{}

// Resource returns an interface for accessing the provided resource.
func (c *FakeMetadataClient) Resource(resource schema.GroupVersionResource) metadata.Getter {
	return &metadataResourceClient{client: c, resource: resource}
}

// Namespace returns an interface for accessing the current resource in the specified
// namespace.
func (c *metadataResourceClient) Namespace(ns string) metadata.ResourceInterface {
	ret := *c
	ret.namespace = ns
	return &ret
}

// CreateFake records the object creation and processes it via the reactor.
func (c *metadataResourceClient) CreateFake(obj *metav1.PartialObjectMetadata, opts metav1.CreateOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
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
	ret, ok := uncastRet.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected return value type %T", uncastRet)
	}
	return ret, err
}

// UpdateFake records the object update and processes it via the reactor.
func (c *metadataResourceClient) UpdateFake(obj *metav1.PartialObjectMetadata, opts metav1.UpdateOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
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
	ret, ok := uncastRet.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected return value type %T", uncastRet)
	}
	return ret, err
}

// UpdateStatus records the object status update and processes it via the reactor.
func (c *metadataResourceClient) UpdateStatus(obj *metav1.PartialObjectMetadata, opts metav1.UpdateOptions) (*metav1.PartialObjectMetadata, error) {
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
	ret, ok := uncastRet.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected return value type %T", uncastRet)
	}
	return ret, err
}

// Delete records the object deletion and processes it via the reactor.
func (c *metadataResourceClient) Delete(ctx context.Context, name string, opts metav1.DeleteOptions, subresources ...string) error {
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		_, err = c.client.Fake.
			Invokes(testing.NewRootDeleteAction(c.resource, name), &metav1.Status{Status: "metadata delete fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		_, err = c.client.Fake.
			Invokes(testing.NewRootDeleteSubresourceAction(c.resource, strings.Join(subresources, "/"), name), &metav1.Status{Status: "metadata delete fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		_, err = c.client.Fake.
			Invokes(testing.NewDeleteAction(c.resource, c.namespace, name), &metav1.Status{Status: "metadata delete fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		_, err = c.client.Fake.
			Invokes(testing.NewDeleteSubresourceAction(c.resource, strings.Join(subresources, "/"), c.namespace, name), &metav1.Status{Status: "metadata delete fail"})
	}

	return err
}

// DeleteCollection records the object collection deletion and processes it via the reactor.
func (c *metadataResourceClient) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	var err error
	switch {
	case len(c.namespace) == 0:
		action := testing.NewRootDeleteCollectionAction(c.resource, listOptions)
		_, err = c.client.Fake.Invokes(action, &metav1.Status{Status: "metadata deletecollection fail"})

	case len(c.namespace) > 0:
		action := testing.NewDeleteCollectionAction(c.resource, c.namespace, listOptions)
		_, err = c.client.Fake.Invokes(action, &metav1.Status{Status: "metadata deletecollection fail"})

	}

	return err
}

// Get records the object retrieval and processes it via the reactor.
func (c *metadataResourceClient) Get(ctx context.Context, name string, opts metav1.GetOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootGetAction(c.resource, name), &metav1.Status{Status: "metadata get fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootGetSubresourceAction(c.resource, strings.Join(subresources, "/"), name), &metav1.Status{Status: "metadata get fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewGetAction(c.resource, c.namespace, name), &metav1.Status{Status: "metadata get fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewGetSubresourceAction(c.resource, c.namespace, strings.Join(subresources, "/"), name), &metav1.Status{Status: "metadata get fail"})
	}

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}
	ret, ok := uncastRet.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected return value type %T", uncastRet)
	}
	return ret, err
}

// List records the object deletion and processes it via the reactor.
func (c *metadataResourceClient) List(ctx context.Context, opts metav1.ListOptions) (*metav1.PartialObjectMetadataList, error) {
	var obj runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0:
		obj, err = c.client.Fake.
			Invokes(testing.NewRootListAction(c.resource, schema.GroupVersionKind{Group: "fake-metadata-client-group", Version: "v1", Kind: "" /*List is appended by the tracker automatically*/}, opts), &metav1.Status{Status: "metadata list fail"})

	case len(c.namespace) > 0:
		obj, err = c.client.Fake.
			Invokes(testing.NewListAction(c.resource, schema.GroupVersionKind{Group: "fake-metadata-client-group", Version: "v1", Kind: "" /*List is appended by the tracker automatically*/}, c.namespace, opts), &metav1.Status{Status: "metadata list fail"})

	}

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}

	inputList, ok := obj.(*metav1.List)
	if !ok {
		return nil, fmt.Errorf("incoming object is incorrect type %T", obj)
	}

	list := &metav1.PartialObjectMetadataList{
		ListMeta: inputList.ListMeta,
	}
	for i := range inputList.Items {
		item, ok := inputList.Items[i].Object.(*metav1.PartialObjectMetadata)
		if !ok {
			return nil, fmt.Errorf("item %d in list %T is %T", i, inputList, inputList.Items[i].Object)
		}
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

func (c *metadataResourceClient) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
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

// Patch records the object patch and processes it via the reactor.
func (c *metadataResourceClient) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
	var uncastRet runtime.Object
	var err error
	switch {
	case len(c.namespace) == 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootPatchAction(c.resource, name, pt, data), &metav1.Status{Status: "metadata patch fail"})

	case len(c.namespace) == 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewRootPatchSubresourceAction(c.resource, name, pt, data, subresources...), &metav1.Status{Status: "metadata patch fail"})

	case len(c.namespace) > 0 && len(subresources) == 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewPatchAction(c.resource, c.namespace, name, pt, data), &metav1.Status{Status: "metadata patch fail"})

	case len(c.namespace) > 0 && len(subresources) > 0:
		uncastRet, err = c.client.Fake.
			Invokes(testing.NewPatchSubresourceAction(c.resource, c.namespace, name, pt, data, subresources...), &metav1.Status{Status: "metadata patch fail"})

	}

	if err != nil {
		return nil, err
	}
	if uncastRet == nil {
		return nil, err
	}
	ret, ok := uncastRet.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected return value type %T", uncastRet)
	}
	return ret, err
}
