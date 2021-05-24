// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import (
	"errors"
	"fmt"
	"log"
	"reflect"
	"strings"

	"sigs.k8s.io/kustomize/api/filters/patchstrategicmerge"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	kyaml "sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/yaml"
)

// Resource is an RNode, representing a Kubernetes Resource Model object,
// paired with metadata used by kustomize.
type Resource struct {
	// TODO: Inline RNode, dropping complexity. Resource is just a decorator.
	node        *kyaml.RNode
	options     *types.GenArgs
	refBy       []resid.ResId
	refVarNames []string
}

const (
	buildAnnotationPreviousKinds      = konfig.ConfigAnnoDomain + "/previousKinds"
	buildAnnotationPreviousNames      = konfig.ConfigAnnoDomain + "/previousNames"
	buildAnnotationPrefixes           = konfig.ConfigAnnoDomain + "/prefixes"
	buildAnnotationSuffixes           = konfig.ConfigAnnoDomain + "/suffixes"
	buildAnnotationPreviousNamespaces = konfig.ConfigAnnoDomain + "/previousNamespaces"

	// the following are only for patches, to specify whether they can change names
	// and kinds of their targets
	buildAnnotationAllowNameChange = konfig.ConfigAnnoDomain + "/allowNameChange"
	buildAnnotationAllowKindChange = konfig.ConfigAnnoDomain + "/allowKindChange"
)

var buildAnnotations = []string{
	buildAnnotationPreviousKinds,
	buildAnnotationPreviousNames,
	buildAnnotationPrefixes,
	buildAnnotationSuffixes,
	buildAnnotationPreviousNamespaces,
	buildAnnotationAllowNameChange,
	buildAnnotationAllowKindChange,
}

func (r *Resource) AsRNode() *kyaml.RNode {
	return r.node.Copy()
}

func (r *Resource) ResetPrimaryData(incoming *Resource) {
	r.node = incoming.node.Copy()
}

func (r *Resource) GetAnnotations() map[string]string {
	annotations, err := r.node.GetAnnotations()
	if err != nil || annotations == nil {
		return make(map[string]string)
	}
	return annotations
}

func (r *Resource) GetFieldValue(f string) (interface{}, error) {
	//nolint:staticcheck
	return r.node.GetFieldValue(f)
}

func (r *Resource) GetDataMap() map[string]string {
	return r.node.GetDataMap()
}

func (r *Resource) GetBinaryDataMap() map[string]string {
	return r.node.GetBinaryDataMap()
}

func (r *Resource) GetGvk() resid.Gvk {
	meta, err := r.node.GetMeta()
	if err != nil {
		return resid.GvkFromString("")
	}
	g, v := resid.ParseGroupVersion(meta.APIVersion)
	return resid.Gvk{Group: g, Version: v, Kind: meta.Kind}
}

func (r *Resource) Hash(h ifc.KustHasher) (string, error) {
	return h.Hash(r.node)
}

func (r *Resource) GetKind() string {
	return r.node.GetKind()
}

func (r *Resource) GetLabels() map[string]string {
	l, err := r.node.GetLabels()
	if err != nil {
		return map[string]string{}
	}
	return l
}

func (r *Resource) GetName() string {
	return r.node.GetName()
}

func (r *Resource) GetSlice(p string) ([]interface{}, error) {
	//nolint:staticcheck
	return r.node.GetSlice(p)
}

func (r *Resource) GetString(p string) (string, error) {
	//nolint:staticcheck
	return r.node.GetString(p)
}

func (r *Resource) IsEmpty() bool {
	return r.node.IsNilOrEmpty()
}

func (r *Resource) Map() (map[string]interface{}, error) {
	return r.node.Map()
}

func (r *Resource) MarshalJSON() ([]byte, error) {
	return r.node.MarshalJSON()
}

func (r *Resource) MatchesLabelSelector(selector string) (bool, error) {
	return r.node.MatchesLabelSelector(selector)
}

func (r *Resource) MatchesAnnotationSelector(selector string) (bool, error) {
	return r.node.MatchesAnnotationSelector(selector)
}

func (r *Resource) SetAnnotations(m map[string]string) {
	if len(m) == 0 {
		// Force field erasure.
		r.node.SetAnnotations(nil)
		return
	}
	r.node.SetAnnotations(m)
}

func (r *Resource) SetDataMap(m map[string]string) {
	r.node.SetDataMap(m)
}

func (r *Resource) SetBinaryDataMap(m map[string]string) {
	r.node.SetBinaryDataMap(m)
}

func (r *Resource) SetGvk(gvk resid.Gvk) {
	r.node.SetMapField(
		kyaml.NewScalarRNode(gvk.Kind), kyaml.KindField)
	r.node.SetMapField(
		kyaml.NewScalarRNode(gvk.ApiVersion()), kyaml.APIVersionField)
}

func (r *Resource) SetLabels(m map[string]string) {
	if len(m) == 0 {
		// Force field erasure.
		r.node.SetLabels(nil)
		return
	}
	r.node.SetLabels(m)
}

func (r *Resource) SetName(n string) {
	r.node.SetName(n)
}

func (r *Resource) SetNamespace(n string) {
	r.node.SetNamespace(n)
}

func (r *Resource) SetKind(k string) {
	gvk := r.GetGvk()
	gvk.Kind = k
	r.SetGvk(gvk)
}

func (r *Resource) UnmarshalJSON(s []byte) error {
	return r.node.UnmarshalJSON(s)
}

// ResCtx is an interface describing the contextual added
// kept kustomize in the context of each Resource object.
// Currently mainly the name prefix and name suffix are added.
type ResCtx interface {
	AddNamePrefix(p string)
	AddNameSuffix(s string)
	GetNamePrefixes() []string
	GetNameSuffixes() []string
}

// ResCtxMatcher returns true if two Resources are being
// modified in the same kustomize context.
type ResCtxMatcher func(ResCtx) bool

// DeepCopy returns a new copy of resource
func (r *Resource) DeepCopy() *Resource {
	rc := &Resource{
		node: r.node.Copy(),
	}
	rc.copyOtherFields(r)
	return rc
}

// CopyMergeMetaDataFields copies everything but the non-metadata in
// the resource.
func (r *Resource) CopyMergeMetaDataFieldsFrom(other *Resource) {
	r.SetLabels(mergeStringMaps(other.GetLabels(), r.GetLabels()))
	r.SetAnnotations(
		mergeStringMaps(other.GetAnnotations(), r.GetAnnotations()))
	r.SetName(other.GetName())
	r.SetNamespace(other.GetNamespace())
	r.copyOtherFields(other)
}

func (r *Resource) copyOtherFields(other *Resource) {
	r.options = other.options
	r.refBy = other.copyRefBy()
	r.refVarNames = copyStringSlice(other.refVarNames)
}

func (r *Resource) MergeDataMapFrom(o *Resource) {
	r.SetDataMap(mergeStringMaps(o.GetDataMap(), r.GetDataMap()))
}

func (r *Resource) MergeBinaryDataMapFrom(o *Resource) {
	r.SetBinaryDataMap(mergeStringMaps(o.GetBinaryDataMap(), r.GetBinaryDataMap()))
}

func (r *Resource) ErrIfNotEquals(o *Resource) error {
	meYaml, err := r.AsYAML()
	if err != nil {
		return err
	}
	otherYaml, err := o.AsYAML()
	if err != nil {
		return err
	}
	if !r.ReferencesEqual(o) {
		return fmt.Errorf(
			`unequal references - self:
%sreferenced by: %s
--- other:
%sreferenced by: %s
`, meYaml, r.GetRefBy(), otherYaml, o.GetRefBy())
	}
	if string(meYaml) != string(otherYaml) {
		return fmt.Errorf(`---  self:
%s
--- other:
%s
`, meYaml, otherYaml)
	}
	return nil
}

func (r *Resource) ReferencesEqual(other *Resource) bool {
	setSelf := make(map[resid.ResId]bool)
	setOther := make(map[resid.ResId]bool)
	for _, ref := range other.refBy {
		setOther[ref] = true
	}
	for _, ref := range r.refBy {
		if _, ok := setOther[ref]; !ok {
			return false
		}
		setSelf[ref] = true
	}
	return len(setSelf) == len(setOther)
}

// NodeEqual returns true if the resource's nodes are
// equal, ignoring ancillary information like genargs, refby, etc.
func (r *Resource) NodeEqual(o *Resource) bool {
	return reflect.DeepEqual(r.node, o.node)
}

func (r *Resource) copyRefBy() []resid.ResId {
	if r.refBy == nil {
		return nil
	}
	s := make([]resid.ResId, len(r.refBy))
	copy(s, r.refBy)
	return s
}

func copyStringSlice(s []string) []string {
	if s == nil {
		return nil
	}
	c := make([]string, len(s))
	copy(c, s)
	return c
}

// Implements ResCtx AddNamePrefix
func (r *Resource) AddNamePrefix(p string) {
	r.appendCsvAnnotation(buildAnnotationPrefixes, p)
}

// Implements ResCtx AddNameSuffix
func (r *Resource) AddNameSuffix(s string) {
	r.appendCsvAnnotation(buildAnnotationSuffixes, s)
}

func (r *Resource) appendCsvAnnotation(name, value string) {
	if value == "" {
		return
	}
	annotations := r.GetAnnotations()
	if existing, ok := annotations[name]; ok {
		annotations[name] = existing + "," + value
	} else {
		annotations[name] = value
	}
	r.SetAnnotations(annotations)
}

func SameEndingSubarray(shortest, longest []string) bool {
	if len(shortest) > len(longest) {
		longest, shortest = shortest, longest
	}
	diff := len(longest) - len(shortest)
	if len(shortest) == 0 {
		return diff == 0
	}
	for i := len(shortest) - 1; i >= 0; i-- {
		if longest[i+diff] != shortest[i] {
			return false
		}
	}
	return true
}

// Implements ResCtx GetNamePrefixes
func (r *Resource) GetNamePrefixes() []string {
	return r.getCsvAnnotation(buildAnnotationPrefixes)
}

// Implements ResCtx GetNameSuffixes
func (r *Resource) GetNameSuffixes() []string {
	return r.getCsvAnnotation(buildAnnotationSuffixes)
}

func (r *Resource) getCsvAnnotation(name string) []string {
	annotations := r.GetAnnotations()
	if _, ok := annotations[name]; !ok {
		return nil
	}
	return strings.Split(annotations[name], ",")
}

// PrefixesSuffixesEquals is conceptually doing the same task
// as OutermostPrefixSuffix but performs a deeper comparison
// of the suffix and prefix slices.
func (r *Resource) PrefixesSuffixesEquals(o ResCtx) bool {
	return SameEndingSubarray(r.GetNamePrefixes(), o.GetNamePrefixes()) && SameEndingSubarray(r.GetNameSuffixes(), o.GetNameSuffixes())
}

// RemoveBuildAnnotations removes annotations created by the build process.
// These are internal-only to kustomize, added to the data pipeline to
// track name changes so name references can be fixed.
func (r *Resource) RemoveBuildAnnotations() {
	annotations := r.GetAnnotations()
	if len(annotations) == 0 {
		return
	}
	for _, a := range buildAnnotations {
		delete(annotations, a)
	}
	r.SetAnnotations(annotations)
}

func (r *Resource) setPreviousId(ns string, n string, k string) *Resource {
	r.appendCsvAnnotation(buildAnnotationPreviousNames, n)
	r.appendCsvAnnotation(buildAnnotationPreviousNamespaces, ns)
	r.appendCsvAnnotation(buildAnnotationPreviousKinds, k)
	return r
}

func (r *Resource) SetAllowNameChange(value string) {
	annotations := r.GetAnnotations()
	annotations[buildAnnotationAllowNameChange] = value
	r.SetAnnotations(annotations)
}

func (r *Resource) NameChangeAllowed() bool {
	annotations := r.GetAnnotations()
	if allowed, set := annotations[buildAnnotationAllowNameChange]; set && allowed == "true" {
		return true
	}
	return false
}

func (r *Resource) SetAllowKindChange(value string) {
	annotations := r.GetAnnotations()
	annotations[buildAnnotationAllowKindChange] = value
	r.SetAnnotations(annotations)
}

func (r *Resource) KindChangeAllowed() bool {
	annotations := r.GetAnnotations()
	if allowed, set := annotations[buildAnnotationAllowKindChange]; set && allowed == "true" {
		return true
	}
	return false
}

// String returns resource as JSON.
func (r *Resource) String() string {
	bs, err := r.MarshalJSON()
	if err != nil {
		return "<" + err.Error() + ">"
	}
	return strings.TrimSpace(string(bs)) + r.options.String()
}

// AsYAML returns the resource in Yaml form.
// Easier to read than JSON.
func (r *Resource) AsYAML() ([]byte, error) {
	json, err := r.MarshalJSON()
	if err != nil {
		return nil, err
	}
	return yaml.JSONToYAML(json)
}

// MustYaml returns YAML or panics.
func (r *Resource) MustYaml() string {
	yml, err := r.AsYAML()
	if err != nil {
		log.Fatal(err)
	}
	return string(yml)
}

// SetOptions updates the generator options for the resource.
func (r *Resource) SetOptions(o *types.GenArgs) {
	r.options = o
}

// Behavior returns the behavior for the resource.
func (r *Resource) Behavior() types.GenerationBehavior {
	return r.options.Behavior()
}

// NeedHashSuffix returns true if a resource content
// hash should be appended to the name of the resource.
func (r *Resource) NeedHashSuffix() bool {
	return r.options != nil && r.options.ShouldAddHashSuffixToName()
}

// GetNamespace returns the namespace the resource thinks it's in.
func (r *Resource) GetNamespace() string {
	namespace, _ := r.GetString("metadata.namespace")
	// if err, namespace is empty, so no need to check.
	return namespace
}

// OrgId returns the original, immutable ResId for the resource.
// This doesn't have to be unique in a ResMap.
func (r *Resource) OrgId() resid.ResId {
	ids := r.PrevIds()
	if len(ids) > 0 {
		return ids[0]
	}
	return r.CurId()
}

// PrevIds returns a list of ResIds that includes every
// previous ResId the resource has had through all of its
// GVKN transformations, in the order that it had that ID.
// I.e. the oldest ID is first.
// The returned array does not include the resource's current
// ID. If there are no previous IDs, this will return nil.
func (r *Resource) PrevIds() []resid.ResId {
	var ids []resid.ResId
	// TODO: merge previous names and namespaces into one list of
	//     pairs on one annotation so there is no chance of error
	names := r.getCsvAnnotation(buildAnnotationPreviousNames)
	ns := r.getCsvAnnotation(buildAnnotationPreviousNamespaces)
	kinds := r.getCsvAnnotation(buildAnnotationPreviousKinds)
	if len(names) != len(ns) || len(names) != len(kinds) {
		panic(errors.New(
			"number of previous names, " +
				"number of previous namespaces, " +
				"number of previous kinds not equal"))
	}
	for i := range names {
		k := kinds[i]
		gvk := r.GetGvk()
		gvk.Kind = k
		ids = append(ids, resid.NewResIdWithNamespace(
			gvk, names[i], ns[i]))
	}
	return ids
}

// StorePreviousId stores the resource's current ID via build annotations.
func (r *Resource) StorePreviousId() {
	id := r.CurId()
	r.setPreviousId(id.EffectiveNamespace(), id.Name, id.Kind)
}

// CurId returns a ResId for the resource using the
// mutable parts of the resource.
// This should be unique in any ResMap.
func (r *Resource) CurId() resid.ResId {
	return resid.NewResIdWithNamespace(
		r.GetGvk(), r.GetName(), r.GetNamespace())
}

// GetRefBy returns the ResIds that referred to current resource
func (r *Resource) GetRefBy() []resid.ResId {
	return r.refBy
}

// AppendRefBy appends a ResId into the refBy list
func (r *Resource) AppendRefBy(id resid.ResId) {
	r.refBy = append(r.refBy, id)
}

// GetRefVarNames returns vars that refer to current resource
func (r *Resource) GetRefVarNames() []string {
	return r.refVarNames
}

// AppendRefVarName appends a name of a var into the refVar list
func (r *Resource) AppendRefVarName(variable types.Var) {
	r.refVarNames = append(r.refVarNames, variable.Name)
}

// ApplySmPatch applies the provided strategic merge patch.
func (r *Resource) ApplySmPatch(patch *Resource) error {
	n, ns, k := r.GetName(), r.GetNamespace(), r.GetKind()
	if patch.NameChangeAllowed() || patch.KindChangeAllowed() {
		r.StorePreviousId()
	}
	if err := r.ApplyFilter(patchstrategicmerge.Filter{
		Patch: patch.node,
	}); err != nil {
		return err
	}
	if r.IsEmpty() {
		return nil
	}
	if !patch.KindChangeAllowed() {
		r.SetKind(k)
	}
	if !patch.NameChangeAllowed() {
		r.SetName(n)
	}
	r.SetNamespace(ns)
	return nil
}

func (r *Resource) ApplyFilter(f kio.Filter) error {
	l, err := f.Filter([]*kyaml.RNode{r.node})
	if len(l) == 0 {
		// The node was deleted.  The following makes r.IsEmpty() true.
		r.node = nil
	}
	return err
}

func mergeStringMaps(maps ...map[string]string) map[string]string {
	result := map[string]string{}
	for _, m := range maps {
		for key, value := range m {
			result[key] = value
		}
	}
	return result
}
