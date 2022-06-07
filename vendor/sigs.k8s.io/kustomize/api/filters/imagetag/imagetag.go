// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package imagetag

import (
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/filters/fsslice"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter modifies an "image tag", the value used to specify the
// name, tag, version digest etc. of (docker) container images
// used by a pod template.
type Filter struct {
	// imageTag is the tag we want to apply to the inputs
	// The name of the image is used as a key, and other fields
	// can specify a new name, tag, etc.
	ImageTag types.Image `json:"imageTag,omitempty" yaml:"imageTag,omitempty"`

	// FsSlice contains the FieldSpecs to locate an image field,
	// e.g. Path: "spec/myContainers[]/image"
	FsSlice types.FsSlice `json:"fieldSpecs,omitempty" yaml:"fieldSpecs,omitempty"`

	trackableSetter filtersutil.TrackableSetter
}

var _ kio.Filter = Filter{}
var _ kio.TrackableFilter = &Filter{}

// WithMutationTracker registers a callback which will be invoked each time a field is mutated
func (f *Filter) WithMutationTracker(callback func(key, value, tag string, node *yaml.RNode)) {
	f.trackableSetter.WithMutationTracker(callback)
}

func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	_, err := kio.FilterAll(yaml.FilterFunc(f.filter)).Filter(nodes)
	return nodes, err
}

func (f Filter) filter(node *yaml.RNode) (*yaml.RNode, error) {
	// FsSlice is an allowlist, not a denyList, so to deny
	// something via configuration a new config mechanism is
	// needed. Until then, hardcode it.
	if f.isOnDenyList(node) {
		return node, nil
	}
	if err := node.PipeE(fsslice.Filter{
		FsSlice: f.FsSlice,
		SetValue: imageTagUpdater{
			ImageTag:        f.ImageTag,
			trackableSetter: f.trackableSetter,
		}.SetImageValue,
	}); err != nil {
		return nil, err
	}
	return node, nil
}

func (f Filter) isOnDenyList(node *yaml.RNode) bool {
	meta, err := node.GetMeta()
	if err != nil {
		// A missing 'meta' field will cause problems elsewhere;
		// ignore it here to keep the signature simple.
		return false
	}
	// Ignore CRDs
	// https://github.com/kubernetes-sigs/kustomize/issues/890
	return meta.Kind == `CustomResourceDefinition`
}
