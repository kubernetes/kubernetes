// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package configmapandsecret

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/types"
)

// Factory makes ConfigMaps and Secrets.
type Factory struct {
	kvLdr   ifc.KvLoader
	options *types.GeneratorOptions
}

// NewFactory returns a new factory that makes ConfigMaps and Secrets.
func NewFactory(
	kvLdr ifc.KvLoader, o *types.GeneratorOptions) *Factory {
	return &Factory{kvLdr: kvLdr, options: o}
}

// setLabelsAndAnnnotations will take the labels and annotations from
// global GeneratorOptions and resource level GeneratorOptions and merge them
// with the resource level taking precedence, and then set them on the provided
// obj.
func (f *Factory) setLabelsAndAnnnotations(obj metav1.Object, opts *types.GeneratorOptions) {
	labels := make(map[string]string)
	annotations := make(map[string]string)
	if f.options != nil {
		for k, v := range f.options.Labels {
			labels[k] = v
		}
		for k, v := range f.options.Annotations {
			annotations[k] = v
		}
	}
	if opts != nil {
		for k, v := range opts.Labels {
			labels[k] = v
		}
		for k, v := range opts.Annotations {
			annotations[k] = v
		}
	}
	if len(labels) != 0 {
		obj.SetLabels(labels)
	}
	if len(annotations) != 0 {
		obj.SetAnnotations(annotations)
	}
}

const keyExistsErrorMsg = "cannot add key %s, another key by that name already exists: %v"
