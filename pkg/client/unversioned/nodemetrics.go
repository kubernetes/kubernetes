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
	"k8s.io/kubernetes/pkg/api"
	exp "k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// NodeMetrics has methods to work with NodeMetrics resources
type NodeMetrics interface {
	NodeMetrics() NodeMetricsInterface
}

// NodeMetricsInterface has methods to work with NodeMetrics resources.
type NodeMetricsInterface interface {
	List(label labels.Selector, field fields.Selector) (*exp.DerivedNodeMetricsList, error)
	Get(name string) (*exp.DerivedNodeMetrics, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(nodeMetrics *exp.DerivedNodeMetrics) (*exp.DerivedNodeMetrics, error)
	Update(nodeMetrics *exp.DerivedNodeMetrics) (*exp.DerivedNodeMetrics, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// nodeMetricsClient implements NodeMetricsInterface
type nodeMetricsClient struct {
	client *ExperimentalClient
}

// newNodeMetrics returns a nodeMetricsClient
func newNodeMetrics(c *ExperimentalClient) *nodeMetricsClient {
	return &nodeMetricsClient{
		client: c,
	}
}

// List takes label and field selectors, and returns the list of
// DerivedNodeMetrics that match those selectors.
func (c *nodeMetricsClient) List(label labels.Selector, field fields.Selector) (result *exp.DerivedNodeMetricsList, err error) {
	result = &exp.DerivedNodeMetricsList{}
	err = c.client.Get().Resource("nodeMetrics").LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return
}

// Get takes the name of the node, and returns the corresponding
// DerivedNodeMetrics object, and an error if it occurs
func (c *nodeMetricsClient) Get(name string) (result *exp.DerivedNodeMetrics, err error) {
	result = &exp.DerivedNodeMetrics{}
	err = c.client.Get().Resource("nodeMetrics").Name(name).Do().Into(result)
	return
}

// Delete takes the name of a node and deletes its DerivedNodeMetrics. Returns
// an error if one occurs.
func (c *nodeMetricsClient) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Resource("nodeMetrics").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion())
	if err != nil {
		return err
	}
	return c.client.Delete().Resource("nodeMetrics").Name(name).Body(body).Do().Error()
}

// Create takes the representation of a nodeMetrics and creates it. Returns
// the server's representation of the nodeMetrics, and an error, if it occurs.
func (c *nodeMetricsClient) Create(nodeMetrics *exp.DerivedNodeMetrics) (result *exp.DerivedNodeMetrics, err error) {
	result = &exp.DerivedNodeMetrics{}
	err = c.client.Post().Resource("nodeMetrics").Body(nodeMetrics).Do().Into(result)
	return
}

// Update takes the representation of a nodeMetrics and updates it. Returns
// the server's representation of the nodeMetrics, and an error, if it occurs.
func (c *nodeMetricsClient) Update(nodeMetrics *exp.DerivedNodeMetrics) (result *exp.DerivedNodeMetrics, err error) {
	result = &exp.DerivedNodeMetrics{}
	err = c.client.Put().Resource("nodeMetrics").Name(nodeMetrics.Name).Body(nodeMetrics).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested DerivedNodeMetrics.
func (c *nodeMetricsClient) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("nodeMetrics").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
