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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	v1alpha1 "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
)

// FakePodMetricses implements PodMetricsInterface
type FakePodMetricses struct {
	Fake *FakeMetricsV1alpha1
	ns   string
}

var podmetricsesResource = schema.GroupVersionResource{Group: "metrics", Version: "v1alpha1", Resource: "podmetricses"}

func (c *FakePodMetricses) Get(name string, options v1.GetOptions) (result *v1alpha1.PodMetrics, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(podmetricsesResource, c.ns, name), &v1alpha1.PodMetrics{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PodMetrics), err
}

func (c *FakePodMetricses) List(opts v1.ListOptions) (result *v1alpha1.PodMetricsList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(podmetricsesResource, c.ns, opts), &v1alpha1.PodMetricsList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.PodMetricsList{}
	for _, item := range obj.(*v1alpha1.PodMetricsList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podMetricses.
func (c *FakePodMetricses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(podmetricsesResource, c.ns, opts))

}
