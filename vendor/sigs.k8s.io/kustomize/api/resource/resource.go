// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import (
	"fmt"
	"log"
	"strings"

	"sigs.k8s.io/kustomize/api/filters/patchstrategicmerge"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/utils"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/resid"
	kyaml "sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/yaml"
)

// Resource is an RNode, representing a Kubernetes Resource Model object,
// paired with metadata used by kustomize.
type Resource struct {
	kyaml.RNode
	options     *types.GenArgs
	refBy       []resid.ResId
	refVarNames []string
}

var BuildAnnotations = []string{
	utils.BuildAnnotationPreviousKinds,
	utils.BuildAnnotationPreviousNames,
	utils.BuildAnnotationPrefixes,
	utils.BuildAnnotationSuffixes,
	utils.BuildAnnotationPreviousNamespaces,
	utils.BuildAnnotationAllowNameChange,
	utils.BuildAnnotationAllowKindChange,
}

func (r *Resource) ResetRNode(incoming *Resource) {
	r.RNode = *incoming.Copy()
}

func (r *Resource) GetGvk() resid.Gvk {
	return resid.GvkFromNode(&r.RNode)
}

func (r *Resource) Hash(h ifc.KustHasher) (string, error) {
	return h.Hash(&r.RNode)
}

func (r *Resource) SetGvk(gvk resid.Gvk) {
	r.SetKind(gvk.Kind)
	r.SetApiVersion(gvk.ApiVersion())
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
		RNode: *r.Copy(),
	}
	rc.copyKustomizeSpecificFields(r)
	return rc
}

// CopyMergeMetaDataFieldsFrom copies everything but the non-metadata in
// the resource.
// TODO: move to RNode, use GetMeta to improve performance.
// Must remove the kustomize bit at the end.
func (r *Resource) CopyMergeMetaDataFieldsFrom(other *Resource) error {
	if err := r.SetLabels(
		mergeStringMaps(other.GetLabels(), r.GetLabels())); err != nil {
		return fmt.Errorf("copyMerge cannot set labels - %w", err)
	}
	if err := r.SetAnnotations(
		mergeStringMaps(other.GetAnnotations(), r.GetAnnotations())); err != nil {
		return fmt.Errorf("copyMerge cannot set annotations - %w", err)
	}
	if err := r.SetName(other.GetName()); err != nil {
		return fmt.Errorf("copyMerge cannot set name - %w", err)
	}
	if err := r.SetNamespace(other.GetNamespace()); err != nil {
		return fmt.Errorf("copyMerge cannot set namespace - %w", err)
	}
	r.copyKustomizeSpecificFields(other)
	return nil
}

func (r *Resource) copyKustomizeSpecificFields(other *Resource) {
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
	r.appendCsvAnnotation(utils.BuildAnnotationPrefixes, p)
}

// Implements ResCtx AddNameSuffix
func (r *Resource) AddNameSuffix(s string) {
	r.appendCsvAnnotation(utils.BuildAnnotationSuffixes, s)
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
	if err := r.SetAnnotations(annotations); err != nil {
		panic(err)
	}
}

// Implements ResCtx GetNamePrefixes
func (r *Resource) GetNamePrefixes() []string {
	return r.getCsvAnnotation(utils.BuildAnnotationPrefixes)
}

// Implements ResCtx GetNameSuffixes
func (r *Resource) GetNameSuffixes() []string {
	return r.getCsvAnnotation(utils.BuildAnnotationSuffixes)
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
	return utils.SameEndingSubSlice(r.GetNamePrefixes(), o.GetNamePrefixes()) &&
		utils.SameEndingSubSlice(r.GetNameSuffixes(), o.GetNameSuffixes())
}

// RemoveBuildAnnotations removes annotations created by the build process.
// These are internal-only to kustomize, added to the data pipeline to
// track name changes so name references can be fixed.
func (r *Resource) RemoveBuildAnnotations() {
	annotations := r.GetAnnotations()
	if len(annotations) == 0 {
		return
	}
	for _, a := range BuildAnnotations {
		delete(annotations, a)
	}
	if err := r.SetAnnotations(annotations); err != nil {
		panic(err)
	}
}

func (r *Resource) setPreviousId(ns string, n string, k string) *Resource {
	r.appendCsvAnnotation(utils.BuildAnnotationPreviousNames, n)
	r.appendCsvAnnotation(utils.BuildAnnotationPreviousNamespaces, ns)
	r.appendCsvAnnotation(utils.BuildAnnotationPreviousKinds, k)
	return r
}

// AllowNameChange allows name changes to the resource.
func (r *Resource) AllowNameChange() {
	annotations := r.GetAnnotations()
	annotations[utils.BuildAnnotationAllowNameChange] = utils.Allowed
	if err := r.SetAnnotations(annotations); err != nil {
		panic(err)
	}
}

func (r *Resource) NameChangeAllowed() bool {
	annotations := r.GetAnnotations()
	v, ok := annotations[utils.BuildAnnotationAllowNameChange]
	return ok && v == utils.Allowed
}

// AllowKindChange allows kind changes to the resource.
func (r *Resource) AllowKindChange() {
	annotations := r.GetAnnotations()
	annotations[utils.BuildAnnotationAllowKindChange] = utils.Allowed
	if err := r.SetAnnotations(annotations); err != nil {
		panic(err)
	}
}

func (r *Resource) KindChangeAllowed() bool {
	annotations := r.GetAnnotations()
	v, ok := annotations[utils.BuildAnnotationAllowKindChange]
	return ok && v == utils.Allowed
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
	prevIds, err := utils.PrevIds(&r.RNode)
	if err != nil {
		// this should never happen
		panic(err)
	}
	return prevIds
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
		Patch: &patch.RNode,
	}); err != nil {
		return err
	}
	if r.IsNilOrEmpty() {
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
	l, err := f.Filter([]*kyaml.RNode{&r.RNode})
	if len(l) == 0 {
		// The node was deleted, which means the entire resource
		// must be deleted.  Signal that via the following:
		r.SetYNode(nil)
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
