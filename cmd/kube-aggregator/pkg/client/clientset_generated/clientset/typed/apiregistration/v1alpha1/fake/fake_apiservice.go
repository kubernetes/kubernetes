/*
Copyright 2017 The Kubernetes Authors.

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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	v1alpha1 "k8s.io/kubernetes/cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1"
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeAPIServices implements APIServiceInterface
type FakeAPIServices struct {
	Fake *FakeApiregistrationV1alpha1
}

var apiservicesResource = schema.GroupVersionResource{Group: "apiregistration.k8s.io", Version: "v1alpha1", Resource: "apiservices"}

func (c *FakeAPIServices) Create(aPIService *v1alpha1.APIService) (result *v1alpha1.APIService, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(apiservicesResource, aPIService), &v1alpha1.APIService{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.APIService), err
}

func (c *FakeAPIServices) Update(aPIService *v1alpha1.APIService) (result *v1alpha1.APIService, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(apiservicesResource, aPIService), &v1alpha1.APIService{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.APIService), err
}

func (c *FakeAPIServices) UpdateStatus(aPIService *v1alpha1.APIService) (*v1alpha1.APIService, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(apiservicesResource, "status", aPIService), &v1alpha1.APIService{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.APIService), err
}

func (c *FakeAPIServices) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(apiservicesResource, name), &v1alpha1.APIService{})
	return err
}

func (c *FakeAPIServices) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(apiservicesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.APIServiceList{})
	return err
}

func (c *FakeAPIServices) Get(name string, options meta_v1.GetOptions) (result *v1alpha1.APIService, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(apiservicesResource, name), &v1alpha1.APIService{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.APIService), err
}

func (c *FakeAPIServices) List(opts v1.ListOptions) (result *v1alpha1.APIServiceList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(apiservicesResource, opts), &v1alpha1.APIServiceList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.APIServiceList{}
	for _, item := range obj.(*v1alpha1.APIServiceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested aPIServices.
func (c *FakeAPIServices) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(apiservicesResource, opts))
}

// Patch applies the patch and returns the patched aPIService.
func (c *FakeAPIServices) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.APIService, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(apiservicesResource, name, data, subresources...), &v1alpha1.APIService{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.APIService), err
}
