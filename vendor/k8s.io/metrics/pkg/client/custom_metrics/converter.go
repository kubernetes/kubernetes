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

package custom_metrics

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/rest"

	cmint "k8s.io/metrics/pkg/apis/custom_metrics"
	cmv1beta1 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
	cmv1beta2 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	"k8s.io/metrics/pkg/client/custom_metrics/scheme"
)

var (
	// MetricVersions is the set of metric versions accepted by the converter.
	MetricVersions = []schema.GroupVersion{
		cmv1beta2.SchemeGroupVersion,
		cmv1beta1.SchemeGroupVersion,
		cmint.SchemeGroupVersion,
	}
)

// MetricConverter knows how to convert between external MetricValue versions.
type MetricConverter struct {
	scheme            *runtime.Scheme
	codecs            serializer.CodecFactory
	internalVersioner runtime.GroupVersioner
}

// NewMetricConverter creates a MetricConverter which knows how to convert objects
// between different versions of the custom metrics api.
func NewMetricConverter() *MetricConverter {
	return &MetricConverter{
		scheme: scheme.Scheme,
		codecs: serializer.NewCodecFactory(scheme.Scheme),
		internalVersioner: runtime.NewMultiGroupVersioner(
			scheme.SchemeGroupVersion,
			schema.GroupKind{Group: cmint.GroupName, Kind: ""},
			schema.GroupKind{Group: cmv1beta1.GroupName, Kind: ""},
			schema.GroupKind{Group: cmv1beta2.GroupName, Kind: ""},
		),
	}
}

// Scheme returns the scheme used by this metric converter.
func (c *MetricConverter) Scheme() *runtime.Scheme {
	return c.scheme
}

// Codecs returns the codecs used by this metric converter
func (c *MetricConverter) Codecs() serializer.CodecFactory {
	return c.codecs
}

// ConvertListOptionsToVersion converts converts a set of MetricListOptions
// to the provided GroupVersion.
func (c *MetricConverter) ConvertListOptionsToVersion(opts *cmint.MetricListOptions, version schema.GroupVersion) (runtime.Object, error) {
	paramObj, err := c.UnsafeConvertToVersionVia(opts, version)
	if err != nil {
		return nil, err
	}
	return paramObj, nil
}

// ConvertResultToVersion converts a Result to the provided GroupVersion
func (c *MetricConverter) ConvertResultToVersion(res rest.Result, gv schema.GroupVersion) (runtime.Object, error) {
	if err := res.Error(); err != nil {
		return nil, err
	}

	metricBytes, err := res.Raw()
	if err != nil {
		return nil, err
	}

	decoder := c.codecs.UniversalDecoder(MetricVersions...)
	rawMetricObj, err := runtime.Decode(decoder, metricBytes)
	if err != nil {
		return nil, err
	}

	metricObj, err := c.UnsafeConvertToVersionVia(rawMetricObj, gv)
	if err != nil {
		return nil, err
	}
	return metricObj, nil
}

// unsafeConvertToVersionVia is like Scheme.UnsafeConvertToVersion, but it does so via an internal version first.
// We use it here to work with the v1beta2 client internally, while preserving backwards compatibility for existing custom metrics adapters
func (c *MetricConverter) UnsafeConvertToVersionVia(obj runtime.Object, externalVersion schema.GroupVersion) (runtime.Object, error) {
	objInt, err := c.scheme.UnsafeConvertToVersion(obj, schema.GroupVersion{Group: externalVersion.Group, Version: runtime.APIVersionInternal})
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object to the internal version: %v", err)
	}

	objExt, err := c.scheme.UnsafeConvertToVersion(objInt, externalVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object back to the external version: %v", err)
	}

	return objExt, err
}
