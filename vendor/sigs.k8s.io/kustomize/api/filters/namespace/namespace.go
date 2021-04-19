// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package namespace

import (
	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/filters/fsslice"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	// Namespace is the namespace to apply to the inputs
	Namespace string `yaml:"namespace,omitempty"`

	// FsSlice contains the FieldSpecs to locate the namespace field
	FsSlice types.FsSlice `json:"fieldSpecs,omitempty" yaml:"fieldSpecs,omitempty"`
}

var _ kio.Filter = Filter{}

func (ns Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(ns.run)).Filter(nodes)
}

// Run runs the filter on a single node rather than a slice
func (ns Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	// hacks for hardcoded types -- :(
	if err := ns.hacks(node); err != nil {
		return nil, err
	}

	// Remove the fieldspecs that are for hardcoded fields.  The fieldspecs
	// exist for backwards compatibility with other implementations
	// of this transformation.
	// This implementation of the namespace transformation
	// Does not use the fieldspecs for implementing cases which
	// require hardcoded logic.
	ns.FsSlice = ns.removeFieldSpecsForHacks(ns.FsSlice)

	// transformations based on data -- :)
	err := node.PipeE(fsslice.Filter{
		FsSlice:    ns.FsSlice,
		SetValue:   filtersutil.SetScalar(ns.Namespace),
		CreateKind: yaml.ScalarNode, // Namespace is a ScalarNode
		CreateTag:  yaml.NodeTagString,
	})
	return node, err
}

// hacks applies the namespace transforms that are hardcoded rather
// than specified through FieldSpecs.
func (ns Filter) hacks(obj *yaml.RNode) error {
	meta, err := obj.GetMeta()
	if err != nil {
		return err
	}

	if err := ns.metaNamespaceHack(obj, meta); err != nil {
		return err
	}

	return ns.roleBindingHack(obj, meta)
}

// metaNamespaceHack is a hack for implementing the namespace transform
// for the metadata.namespace field on namespace scoped resources.
// namespace scoped resources are determined by NOT being present
// in a hard-coded list of cluster-scoped resource types (by apiVersion and kind).
//
// This hack should be updated to allow individual resources to specify
// if they are cluster scoped through either an annotation on the resources,
// or through inlined OpenAPI on the resource as a YAML comment.
func (ns Filter) metaNamespaceHack(obj *yaml.RNode, meta yaml.ResourceMeta) error {
	gvk := fieldspec.GetGVK(meta)
	if !gvk.IsNamespaceableKind() {
		return nil
	}
	f := fsslice.Filter{
		FsSlice: []types.FieldSpec{
			{Path: types.MetadataNamespacePath, CreateIfNotPresent: true},
		},
		SetValue:   filtersutil.SetScalar(ns.Namespace),
		CreateKind: yaml.ScalarNode, // Namespace is a ScalarNode
	}
	_, err := f.Filter(obj)
	return err
}

// roleBindingHack is a hack for implementing the namespace transform
// for RoleBinding and ClusterRoleBinding resource types.
// RoleBinding and ClusterRoleBinding have namespace set on
// elements of the "subjects" field if and only if the subject elements
// "name" is "default".  Otherwise the namespace is not set.
//
// Example:
//
// kind: RoleBinding
// subjects:
// - name: "default" # this will have the namespace set
//   ...
// - name: "something-else" # this will not have the namespace set
//   ...
func (ns Filter) roleBindingHack(obj *yaml.RNode, meta yaml.ResourceMeta) error {
	if meta.Kind != roleBindingKind && meta.Kind != clusterRoleBindingKind {
		return nil
	}

	// Lookup the namespace field on all elements.
	// We should change the fieldspec so this isn't necessary.
	obj, err := obj.Pipe(yaml.Lookup(subjectsField))
	if err != nil || yaml.IsMissingOrNull(obj) {
		return err
	}

	// add the namespace to each "subject" with name: default
	err = obj.VisitElements(func(o *yaml.RNode) error {
		// The only case we need to force the namespace
		// if for the "service account". "default" is
		// kind of hardcoded here for right now.
		name, err := o.Pipe(
			yaml.Lookup("name"), yaml.Match("default"),
		)
		if err != nil || yaml.IsMissingOrNull(name) {
			return err
		}

		// set the namespace for the default account
		v := yaml.NewScalarRNode(ns.Namespace)
		return o.PipeE(
			yaml.LookupCreate(yaml.ScalarNode, "namespace"),
			yaml.FieldSetter{Value: v},
		)
	})

	return err
}

// removeFieldSpecsForHacks removes from the list fieldspecs that
// have hardcoded implementations
func (ns Filter) removeFieldSpecsForHacks(fs types.FsSlice) types.FsSlice {
	var val types.FsSlice
	for i := range fs {
		// implemented by metaNamespaceHack
		if fs[i].Path == types.MetadataNamespacePath {
			continue
		}
		// implemented by roleBindingHack
		if fs[i].Kind == roleBindingKind && fs[i].Path == subjectsField {
			continue
		}
		// implemented by roleBindingHack
		if fs[i].Kind == clusterRoleBindingKind && fs[i].Path == subjectsField {
			continue
		}
		val = append(val, fs[i])
	}
	return val
}

const (
	subjectsField          = "subjects"
	roleBindingKind        = "RoleBinding"
	clusterRoleBindingKind = "ClusterRoleBinding"
)
