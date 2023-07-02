// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package namespace

import (
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/filters/fsslice"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	// Namespace is the namespace to apply to the inputs
	Namespace string `yaml:"namespace,omitempty"`

	// FsSlice contains the FieldSpecs to locate the namespace field
	FsSlice types.FsSlice `json:"fieldSpecs,omitempty" yaml:"fieldSpecs,omitempty"`

	// UnsetOnly means only blank namespace fields will be set
	UnsetOnly bool `json:"unsetOnly" yaml:"unsetOnly"`

	// SetRoleBindingSubjects determines which subject fields in RoleBinding and ClusterRoleBinding
	// objects will have their namespace fields set. Overrides field specs provided for these types, if any.
	// - defaultOnly (default): namespace will be set only on subjects named "default".
	// - allServiceAccounts: namespace will be set on all subjects with "kind: ServiceAccount"
	// - none: all subjects will be skipped.
	SetRoleBindingSubjects RoleBindingSubjectMode `json:"setRoleBindingSubjects" yaml:"setRoleBindingSubjects"`

	trackableSetter filtersutil.TrackableSetter
}

type RoleBindingSubjectMode string

const (
	DefaultSubjectsOnly       RoleBindingSubjectMode = "defaultOnly"
	SubjectModeUnspecified    RoleBindingSubjectMode = ""
	AllServiceAccountSubjects RoleBindingSubjectMode = "allServiceAccounts"
	NoSubjects                RoleBindingSubjectMode = "none"
)

var _ kio.Filter = Filter{}
var _ kio.TrackableFilter = &Filter{}

// WithMutationTracker registers a callback which will be invoked each time a field is mutated
func (ns *Filter) WithMutationTracker(callback func(key, value, tag string, node *yaml.RNode)) {
	ns.trackableSetter.WithMutationTracker(callback)
}

func (ns Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(ns.run)).Filter(nodes)
}

// Run runs the filter on a single node rather than a slice
func (ns Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	// Special handling for metadata.namespace and metadata.name -- :(
	// never let SetEntry handle metadata.namespace--it will incorrectly include cluster-scoped resources
	// only update metadata.name if api version is expected one--so-as it leaves other resources of kind namespace alone
	apiVersion := node.GetApiVersion()
	ns.FsSlice = ns.removeUnneededMetaFieldSpecs(apiVersion, ns.FsSlice)
	gvk := resid.GvkFromNode(node)
	if err := ns.metaNamespaceHack(node, gvk); err != nil {
		return nil, err
	}

	// Special handling for (cluster) role binding subjects -- :(
	if isRoleBinding(gvk.Kind) {
		ns.FsSlice = ns.removeRoleBindingSubjectFieldSpecs(ns.FsSlice)
		if err := ns.roleBindingHack(node); err != nil {
			return nil, err
		}
	}

	// transformations based on data -- :)
	err := node.PipeE(fsslice.Filter{
		FsSlice:    ns.FsSlice,
		SetValue:   ns.fieldSetter(),
		CreateKind: yaml.ScalarNode, // Namespace is a ScalarNode
		CreateTag:  yaml.NodeTagString,
	})
	invalidKindErr := &yaml.InvalidNodeKindError{}
	if err != nil && errors.As(err, &invalidKindErr) && invalidKindErr.ActualNodeKind() != yaml.ScalarNode {
		return nil, errors.WrapPrefixf(err, "namespace field specs must target scalar nodes")
	}
	return node, errors.WrapPrefixf(err, "namespace transformation failed")
}

// metaNamespaceHack is a hack for implementing the namespace transform
// for the metadata.namespace field on namespace scoped resources.
func (ns Filter) metaNamespaceHack(obj *yaml.RNode, gvk resid.Gvk) error {
	if gvk.IsClusterScoped() {
		return nil
	}
	f := fsslice.Filter{
		FsSlice: []types.FieldSpec{
			{Path: types.MetadataNamespacePath, CreateIfNotPresent: true},
		},
		SetValue:   ns.fieldSetter(),
		CreateKind: yaml.ScalarNode, // Namespace is a ScalarNode
	}
	_, err := f.Filter(obj)
	return err
}

// roleBindingHack is a hack for implementing the transformer's SetRoleBindingSubjects option
// for RoleBinding and ClusterRoleBinding resource types.
//
// In NoSubjects mode, it does nothing.
//
// In AllServiceAccountSubjects mode, it sets the namespace on subjects with "kind: ServiceAccount".
//
// In DefaultSubjectsOnly mode (default mode), RoleBinding and ClusterRoleBinding have namespace set on
// elements of the "subjects" field if and only if the subject elements
// "name" is "default".  Otherwise the namespace is not set.
// Example:
//
// kind: RoleBinding
// subjects:
// - name: "default" # this will have the namespace set
//   ...
// - name: "something-else" # this will not have the namespace set
//   ...
func (ns Filter) roleBindingHack(obj *yaml.RNode) error {
	var visitor filtersutil.SetFn
	switch ns.SetRoleBindingSubjects {
	case NoSubjects:
		return nil
	case DefaultSubjectsOnly, SubjectModeUnspecified:
		visitor = ns.setSubjectsNamedDefault
	case AllServiceAccountSubjects:
		visitor = ns.setServiceAccountNamespaces
	default:
		return errors.Errorf("invalid value %q for setRoleBindingSubjects: "+
			"must be one of %q, %q or %q", ns.SetRoleBindingSubjects,
			DefaultSubjectsOnly, NoSubjects, AllServiceAccountSubjects)
	}

	// Lookup the subjects field on all elements.
	obj, err := obj.Pipe(yaml.Lookup(subjectsField))
	if err != nil || yaml.IsMissingOrNull(obj) {
		return err
	}
	// Use the appropriate visitor to set the namespace field on the correct subset of subjects
	return errors.WrapPrefixf(obj.VisitElements(visitor), "setting namespace on (cluster)role binding subjects")
}

func isRoleBinding(kind string) bool {
	return kind == roleBindingKind || kind == clusterRoleBindingKind
}

func (ns Filter) setServiceAccountNamespaces(o *yaml.RNode) error {
	name, err := o.Pipe(yaml.Lookup("kind"), yaml.Match("ServiceAccount"))
	if err != nil || yaml.IsMissingOrNull(name) {
		return errors.WrapPrefixf(err, "looking up kind on (cluster)role binding subject")
	}
	return setNamespaceField(o, ns.fieldSetter())
}

func (ns Filter) setSubjectsNamedDefault(o *yaml.RNode) error {
	name, err := o.Pipe(yaml.Lookup("name"), yaml.Match("default"))
	if err != nil || yaml.IsMissingOrNull(name) {
		return errors.WrapPrefixf(err, "looking up name on (cluster)role binding subject")
	}
	return setNamespaceField(o, ns.fieldSetter())
}

func setNamespaceField(node *yaml.RNode, setter filtersutil.SetFn) error {
	node, err := node.Pipe(yaml.LookupCreate(yaml.ScalarNode, "namespace"))
	if err != nil {
		return errors.WrapPrefixf(err, "setting namespace field on (cluster)role binding subject")
	}
	return setter(node)
}

// removeRoleBindingSubjectFieldSpecs removes from the list fieldspecs that
// have hardcoded implementations
func (ns Filter) removeRoleBindingSubjectFieldSpecs(fs types.FsSlice) types.FsSlice {
	var val types.FsSlice
	for i := range fs {
		if isRoleBinding(fs[i].Kind) && fs[i].Path == subjectsNamespacePath {
			continue
		}
		val = append(val, fs[i])
	}
	return val
}

func (ns Filter) removeUnneededMetaFieldSpecs(apiVersion string, fs types.FsSlice) types.FsSlice {
	var val types.FsSlice
	for i := range fs {
		if fs[i].Path == types.MetadataNamespacePath {
			continue
		}
		if apiVersion != types.MetadataNamespaceApiVersion && fs[i].Path == types.MetadataNamePath {
			continue
		}
		val = append(val, fs[i])
	}
	return val
}

func (ns *Filter) fieldSetter() filtersutil.SetFn {
	if ns.UnsetOnly {
		return ns.trackableSetter.SetEntryIfEmpty("", ns.Namespace, yaml.NodeTagString)
	}
	return ns.trackableSetter.SetEntry("", ns.Namespace, yaml.NodeTagString)
}

const (
	subjectsField          = "subjects"
	subjectsNamespacePath  = "subjects/namespace"
	roleBindingKind        = "RoleBinding"
	clusterRoleBindingKind = "ClusterRoleBinding"
)
