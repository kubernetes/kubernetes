/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	testgroup "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testdata/apis/testgroup"
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// TestTypeNamespacer has methods to work with TestType resources in a namespace
type TestTypeNamespacer interface {
	TestTypes(namespace string) TestTypeInterface
}

// TestTypeInterface has methods to work with TestType resources.
type TestTypeInterface interface {
	Create(*testgroup.TestType) (*testgroup.TestType, error)
	Update(*testgroup.TestType) (*testgroup.TestType, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*testgroup.TestType, error)
	List(opts unversioned.ListOptions) (*testgroup.TestTypeList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// testTypes implements TestTypeInterface
type testTypes struct {
	client *TestgroupClient
	ns     string
}

// newTestTypes returns a TestTypes
func newTestTypes(c *TestgroupClient, namespace string) *testTypes {
	return &testTypes{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a testType and creates it.  Returns the server's representation of the testType, and an error, if there is any.
func (c *testTypes) Create(testType *testgroup.TestType) (result *testgroup.TestType, err error) {
	result = &testgroup.TestType{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("testTypes").
		Body(testType).
		Do().
		Into(result)
	return
}

// Update takes the representation of a testType and updates it. Returns the server's representation of the testType, and an error, if there is any.
func (c *testTypes) Update(testType *testgroup.TestType) (result *testgroup.TestType, err error) {
	result = &testgroup.TestType{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("testTypes").
		Name(testType.Name).
		Body(testType).
		Do().
		Into(result)
	return
}

// Delete takes name of the testType and deletes it. Returns an error if one occurs.
func (c *testTypes) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("testTypes").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("testTypes").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the testType, and returns the corresponding testType object, and an error if there is any.
func (c *testTypes) Get(name string) (result *testgroup.TestType, err error) {
	result = &testgroup.TestType{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("testTypes").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of TestTypes that match those selectors.
func (c *testTypes) List(opts unversioned.ListOptions) (result *testgroup.TestTypeList, err error) {
	result = &testgroup.TestTypeList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("testTypes").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested testTypes.
func (c *testTypes) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("testTypes").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
