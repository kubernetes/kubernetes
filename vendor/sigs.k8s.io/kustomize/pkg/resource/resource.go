// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package resource implements representations of k8s API resources as "unstructured" objects.
package resource

import (
	"reflect"
	"strings"

	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

// Resource is map representation of a Kubernetes API resource object
// paired with a GenerationBehavior.
type Resource struct {
	ifc.Kunstructured
	originalName string
	originalNs   string
	options      *types.GenArgs
	refBy        []resid.ResId
	refVarNames  []string
	namePrefixes []string
	nameSuffixes []string
}

// DeepCopy returns a new copy of resource
func (r *Resource) DeepCopy() *Resource {
	rc := &Resource{
		Kunstructured: r.Kunstructured.Copy(),
	}
	rc.copyOtherFields(r)
	return rc
}

// Replace performs replace with other resource.
func (r *Resource) Replace(other *Resource) {
	r.SetLabels(mergeStringMaps(other.GetLabels(), r.GetLabels()))
	r.SetAnnotations(
		mergeStringMaps(other.GetAnnotations(), r.GetAnnotations()))
	r.SetName(other.GetName())
	r.copyOtherFields(other)
}

func (r *Resource) copyOtherFields(other *Resource) {
	r.originalName = other.originalName
	r.originalNs = other.originalNs
	r.options = other.options
	r.refBy = other.copyRefBy()
	r.refVarNames = copyStringSlice(other.refVarNames)
	r.namePrefixes = copyStringSlice(other.namePrefixes)
	r.nameSuffixes = copyStringSlice(other.nameSuffixes)
}

func (r *Resource) Equals(o *Resource) bool {
	return r.ReferencesEqual(o) &&
		reflect.DeepEqual(r.Kunstructured, o.Kunstructured)
}

func (r *Resource) ReferencesEqual(o *Resource) bool {
	setSelf := make(map[resid.ResId]bool)
	setOther := make(map[resid.ResId]bool)
	for _, ref := range o.refBy {
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

func (r *Resource) KunstructEqual(o *Resource) bool {
	return reflect.DeepEqual(r.Kunstructured, o.Kunstructured)
}

// Merge performs merge with other resource.
func (r *Resource) Merge(other *Resource) {
	r.Replace(other)
	mergeConfigmap(r.Map(), other.Map(), r.Map())
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

func (r *Resource) AddNamePrefix(p string) {
	r.namePrefixes = append(r.namePrefixes, p)
}

func (r *Resource) AddNameSuffix(s string) {
	r.nameSuffixes = append(r.nameSuffixes, s)
}

func (r *Resource) GetOutermostNamePrefix() string {
	if len(r.namePrefixes) == 0 {
		return ""
	}
	return r.namePrefixes[len(r.namePrefixes)-1]
}

func (r *Resource) GetOutermostNameSuffix() string {
	if len(r.nameSuffixes) == 0 {
		return ""
	}
	return r.nameSuffixes[len(r.nameSuffixes)-1]
}

func (r *Resource) InSameFuzzyNamespace(o *Resource) bool {
	return r.GetNamespace() == o.GetNamespace() &&
		r.GetOutermostNamePrefix() == o.GetOutermostNamePrefix() &&
		r.GetOutermostNameSuffix() == o.GetOutermostNameSuffix()
}

func (r *Resource) GetOriginalName() string {
	return r.originalName
}

// Making this public would be bad.
func (r *Resource) setOriginalName(n string) *Resource {
	r.originalName = n
	return r
}

func (r *Resource) GetOriginalNs() string {
	return r.originalNs
}

// Making this public would be bad.
func (r *Resource) setOriginalNs(n string) *Resource {
	r.originalNs = n
	return r
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

// Behavior returns the behavior for the resource.
func (r *Resource) Behavior() types.GenerationBehavior {
	return r.options.Behavior()
}

// NeedHashSuffix checks if the resource need a hash suffix
func (r *Resource) NeedHashSuffix() bool {
	return r.options != nil && r.options.NeedsHashSuffix()
}

// GetNamespace returns the namespace the resource thinks it's in.
func (r *Resource) GetNamespace() string {
	namespace, _ := r.GetString("metadata.namespace")
	// if err, namespace is empty, so no need to check.
	return namespace
}

// OrgId returns the original, immutable ResId for the resource.
// This doesn't have to be unique in a ResMap.
// TODO: compute this once and save it in the resource.
func (r *Resource) OrgId() resid.ResId {
	return resid.NewResIdWithNamespace(
		r.GetGvk(), r.GetOriginalName(), r.GetOriginalNs())
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

// TODO: Add BinaryData once we sync to new k8s.io/api
func mergeConfigmap(
	mergedTo map[string]interface{},
	maps ...map[string]interface{}) {
	mergedMap := map[string]interface{}{}
	for _, m := range maps {
		datamap, ok := m["data"].(map[string]interface{})
		if ok {
			for key, value := range datamap {
				mergedMap[key] = value
			}
		}
	}
	mergedTo["data"] = mergedMap
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
