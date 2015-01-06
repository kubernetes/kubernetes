/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type BoundPodsNamespacer interface {
	BoundPods() BoundPodsInterface
}

type BoundPodsInterface interface {
	Get(node string) (result *api.BoundPods, err error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// BoundPodsByNode exposes a method for watching the bound pods for a node.
type BoundPodsByNode struct {
	BoundPodsInterface
}

// WatchNodeBoundPods returns a watch for the bound pods on a node from a given resource version.
func (n BoundPodsByNode) WatchNodeBoundPods(node string, resourceVersion string) (watch.Interface, error) {
	return n.Watch(labels.Everything(), labels.SelectorFromSet(labels.Set{"host": node}), resourceVersion)
}

// boundPods implements BoundPodsInterface interface
type boundPods struct {
	r *Client
}

// newBoundPods returns a boundPods object
func newBoundPods(c *Client) *boundPods {
	return &boundPods{c}
}

// Get gets the bound pods for a node
func (c *boundPods) Get(node string) (*api.BoundPods, error) {
	result := &api.BoundPods{}
	err := c.r.Get().Resource("boundPods").Name(node).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested bound pods. The "host" field selector is supported.
func (c *boundPods) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Resource("boundPods").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
