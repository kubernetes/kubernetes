/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/watch"
)

// StatefulSetNamespacer has methods to work with StatefulSet resources in a namespace
type StatefulSetNamespacer interface {
	StatefulSets(namespace string) StatefulSetInterface
}

// StatefulSetInterface exposes methods to work on StatefulSet resources.
type StatefulSetInterface interface {
	List(opts api.ListOptions) (*apps.StatefulSetList, error)
	Get(name string) (*apps.StatefulSet, error)
	Create(statefulSet *apps.StatefulSet) (*apps.StatefulSet, error)
	Update(statefulSet *apps.StatefulSet) (*apps.StatefulSet, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(statefulSet *apps.StatefulSet) (*apps.StatefulSet, error)
}

// statefulSet implements StatefulSetNamespacer interface
type statefulSet struct {
	r  *AppsClient
	ns string
}

// newStatefulSet returns a statefulSet
func newStatefulSet(c *AppsClient, namespace string) *statefulSet {
	return &statefulSet{c, namespace}
}

// List returns a list of statefulSet that match the label and field selectors.
func (c *statefulSet) List(opts api.ListOptions) (result *apps.StatefulSetList, err error) {
	result = &apps.StatefulSetList{}
	err = c.r.Get().Namespace(c.ns).Resource("statefulsets").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular statefulSet.
func (c *statefulSet) Get(name string) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.r.Get().Namespace(c.ns).Resource("statefulsets").Name(name).Do().Into(result)
	return
}

// Create creates a new statefulSet.
func (c *statefulSet) Create(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.r.Post().Namespace(c.ns).Resource("statefulsets").Body(statefulSet).Do().Into(result)
	return
}

// Update updates an existing statefulSet.
func (c *statefulSet) Update(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.r.Put().Namespace(c.ns).Resource("statefulsets").Name(statefulSet.Name).Body(statefulSet).Do().Into(result)
	return
}

// Delete deletes a statefulSet, returns error if one occurs.
func (c *statefulSet) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("statefulsets").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested statefulSet.
func (c *statefulSet) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("statefulsets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the statefulSet and the new status.  Returns the server's representation of the statefulSet, and an error, if it occurs.
func (c *statefulSet) UpdateStatus(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	result = &apps.StatefulSet{}
	err = c.r.Put().Namespace(c.ns).Resource("statefulsets").Name(statefulSet.Name).SubResource("status").Body(statefulSet).Do().Into(result)
	return
}
