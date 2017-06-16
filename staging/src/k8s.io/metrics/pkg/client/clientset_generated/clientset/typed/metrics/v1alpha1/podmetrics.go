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

package v1alpha1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	v1alpha1 "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	scheme "k8s.io/metrics/pkg/client/clientset_generated/clientset/scheme"
)

// PodMetricsesGetter has a method to return a PodMetricsInterface.
// A group's client should implement this interface.
type PodMetricsesGetter interface {
	PodMetricses(namespace string) PodMetricsInterface
}

// PodMetricsInterface has methods to work with PodMetrics resources.
type PodMetricsInterface interface {
	Get(name string, options v1.GetOptions) (*v1alpha1.PodMetrics, error)
	List(opts v1.ListOptions) (*v1alpha1.PodMetricsList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	PodMetricsExpansion
}

// podMetricses implements PodMetricsInterface
type podMetricses struct {
	client rest.Interface
	ns     string
}

// newPodMetricses returns a PodMetricses
func newPodMetricses(c *MetricsV1alpha1Client, namespace string) *podMetricses {
	return &podMetricses{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the podMetrics, and returns the corresponding podMetrics object, and an error if there is any.
func (c *podMetricses) Get(name string, options v1.GetOptions) (result *v1alpha1.PodMetrics, err error) {
	result = &v1alpha1.PodMetrics{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PodMetricses that match those selectors.
func (c *podMetricses) List(opts v1.ListOptions) (result *v1alpha1.PodMetricsList, err error) {
	result = &v1alpha1.PodMetricsList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podMetricses.
func (c *podMetricses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("pods").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}
