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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/testing"
	"k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	eclient "k8s.io/metrics/pkg/client/external_metrics"
)

type FakeExternalMetricsClient struct {
	testing.Fake
}

func (c *FakeExternalMetricsClient) NamespacedMetrics(namespace string) eclient.MetricsInterface {
	return &fakeNamespacedMetrics{
		Fake: c,
		ns:   namespace,
	}
}

type fakeNamespacedMetrics struct {
	Fake *FakeExternalMetricsClient
	ns   string
}

func (m *fakeNamespacedMetrics) List(metricName string, metricSelector labels.Selector) (*v1beta1.ExternalMetricValueList, error) {
	resource := schema.GroupResource{
		Group:    v1beta1.SchemeGroupVersion.Group,
		Resource: metricName,
	}
	kind := schema.GroupKind{
		Group: v1beta1.SchemeGroupVersion.Group,
		Kind:  "ExternalMetricValue",
	}
	action := testing.NewListAction(resource.WithVersion(""), kind.WithVersion(""), m.ns, metav1.ListOptions{LabelSelector: metricSelector.String()})
	obj, err := m.Fake.
		Invokes(action, &v1beta1.ExternalMetricValueList{})

	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.ExternalMetricValueList), err
}
