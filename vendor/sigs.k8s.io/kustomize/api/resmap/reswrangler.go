// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmap

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	kyaml_yaml "sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/yaml"
)

// resWrangler implements ResMap.
type resWrangler struct {
	// Resource list maintained in load (append) order.
	// This is important for transformers, which must
	// be performed in a specific order, and for users
	// who for whatever reasons wish the order they
	// specify in kustomizations to be maintained and
	// available as an option for final YAML rendering.
	rList []*resource.Resource
}

func newOne() *resWrangler {
	result := &resWrangler{}
	result.Clear()
	return result
}

// Clear implements ResMap.
func (m *resWrangler) Clear() {
	m.rList = nil
}

// Size implements ResMap.
func (m *resWrangler) Size() int {
	return len(m.rList)
}

func (m *resWrangler) indexOfResource(other *resource.Resource) int {
	for i, r := range m.rList {
		if r == other {
			return i
		}
	}
	return -1
}

// Resources implements ResMap.
func (m *resWrangler) Resources() []*resource.Resource {
	tmp := make([]*resource.Resource, len(m.rList))
	copy(tmp, m.rList)
	return tmp
}

// Append implements ResMap.
func (m *resWrangler) Append(res *resource.Resource) error {
	id := res.CurId()
	if r := m.GetMatchingResourcesByCurrentId(id.Equals); len(r) > 0 {
		return fmt.Errorf(
			"may not add resource with an already registered id: %s", id)
	}
	m.rList = append(m.rList, res)
	return nil
}

// Remove implements ResMap.
func (m *resWrangler) Remove(adios resid.ResId) error {
	tmp := newOne()
	for _, r := range m.rList {
		if r.CurId() != adios {
			tmp.Append(r)
		}
	}
	if tmp.Size() != m.Size()-1 {
		return fmt.Errorf("id %s not found in removal", adios)
	}
	m.rList = tmp.rList
	return nil
}

// Replace implements ResMap.
func (m *resWrangler) Replace(res *resource.Resource) (int, error) {
	id := res.CurId()
	i, err := m.GetIndexOfCurrentId(id)
	if err != nil {
		return -1, errors.Wrap(err, "in Replace")
	}
	if i < 0 {
		return -1, fmt.Errorf("cannot find resource with id %s to replace", id)
	}
	m.rList[i] = res
	return i, nil
}

// AllIds implements ResMap.
func (m *resWrangler) AllIds() (ids []resid.ResId) {
	ids = make([]resid.ResId, m.Size())
	for i, r := range m.rList {
		ids[i] = r.CurId()
	}
	return
}

// Debug implements ResMap.
func (m *resWrangler) Debug(title string) {
	fmt.Println("--------------------------- " + title)
	firstObj := true
	for i, r := range m.rList {
		if firstObj {
			firstObj = false
		} else {
			fmt.Println("---")
		}
		fmt.Printf("# %d  %s\n", i, r.OrgId())
		m, err := r.Map()
		if err != nil {
			panic(err)
		}
		blob, err := yaml.Marshal(m)
		if err != nil {
			panic(err)
		}
		fmt.Println(string(blob))
	}
}

type IdMatcher func(resid.ResId) bool

// GetByIndex implements ResMap.
func (m *resWrangler) GetByIndex(i int) *resource.Resource {
	if i < 0 || i >= m.Size() {
		return nil
	}
	return m.rList[i]
}

// GetIndexOfCurrentId implements ResMap.
func (m *resWrangler) GetIndexOfCurrentId(id resid.ResId) (int, error) {
	count := 0
	result := -1
	for i, r := range m.rList {
		if id.Equals(r.CurId()) {
			count++
			result = i
		}
	}
	if count > 1 {
		return -1, fmt.Errorf("id matched %d resources", count)
	}
	return result, nil
}

type IdFromResource func(r *resource.Resource) resid.ResId

func GetCurrentId(r *resource.Resource) resid.ResId { return r.CurId() }

// GetMatchingResourcesByCurrentId implements ResMap.
func (m *resWrangler) GetMatchingResourcesByCurrentId(
	matches IdMatcher) []*resource.Resource {
	return m.filteredById(matches, GetCurrentId)
}

// GetMatchingResourcesByAnyId implements ResMap.
func (m *resWrangler) GetMatchingResourcesByAnyId(
	matches IdMatcher) []*resource.Resource {
	var result []*resource.Resource
	for _, r := range m.rList {
		for _, id := range append(r.PrevIds(), r.CurId()) {
			if matches(id) {
				result = append(result, r)
				break
			}
		}
	}
	return result
}

func (m *resWrangler) filteredById(
	matches IdMatcher, idGetter IdFromResource) []*resource.Resource {
	var result []*resource.Resource
	for _, r := range m.rList {
		if matches(idGetter(r)) {
			result = append(result, r)
		}
	}
	return result
}

// GetByCurrentId implements ResMap.
func (m *resWrangler) GetByCurrentId(
	id resid.ResId) (*resource.Resource, error) {
	return demandOneMatch(m.GetMatchingResourcesByCurrentId, id, "Current")
}

// GetById implements ResMap.
func (m *resWrangler) GetById(
	id resid.ResId) (*resource.Resource, error) {
	r, err := demandOneMatch(m.GetMatchingResourcesByAnyId, id, "Id")
	if err != nil {
		return nil, fmt.Errorf(
			"%s; failed to find unique target for patch %s",
			err.Error(), id.GvknString())
	}
	return r, nil
}

type resFinder func(IdMatcher) []*resource.Resource

func demandOneMatch(
	f resFinder, id resid.ResId, s string) (*resource.Resource, error) {
	r := f(id.Equals)
	if len(r) == 1 {
		return r[0], nil
	}
	if len(r) > 1 {
		return nil, fmt.Errorf("multiple matches for %s %s", s, id)
	}
	return nil, fmt.Errorf("no matches for %sId %s", s, id)
}

// GroupedByCurrentNamespace implements ResMap.GroupByCurrentNamespace
func (m *resWrangler) GroupedByCurrentNamespace() map[string][]*resource.Resource {
	items := m.groupedByCurrentNamespace()
	delete(items, resid.TotallyNotANamespace)
	return items
}

// NonNamespaceable implements ResMap.NonNamespaceable
func (m *resWrangler) NonNamespaceable() []*resource.Resource {
	return m.groupedByCurrentNamespace()[resid.TotallyNotANamespace]
}

func (m *resWrangler) groupedByCurrentNamespace() map[string][]*resource.Resource {
	byNamespace := make(map[string][]*resource.Resource)
	for _, res := range m.rList {
		namespace := res.CurId().EffectiveNamespace()
		if _, found := byNamespace[namespace]; !found {
			byNamespace[namespace] = []*resource.Resource{}
		}
		byNamespace[namespace] = append(byNamespace[namespace], res)
	}
	return byNamespace
}

// GroupedByNamespace implements ResMap.GroupByOrginalNamespace
func (m *resWrangler) GroupedByOriginalNamespace() map[string][]*resource.Resource {
	items := m.groupedByOriginalNamespace()
	delete(items, resid.TotallyNotANamespace)
	return items
}

func (m *resWrangler) groupedByOriginalNamespace() map[string][]*resource.Resource {
	byNamespace := make(map[string][]*resource.Resource)
	for _, res := range m.rList {
		namespace := res.OrgId().EffectiveNamespace()
		if _, found := byNamespace[namespace]; !found {
			byNamespace[namespace] = []*resource.Resource{}
		}
		byNamespace[namespace] = append(byNamespace[namespace], res)
	}
	return byNamespace
}

// AsYaml implements ResMap.
func (m *resWrangler) AsYaml() ([]byte, error) {
	firstObj := true
	var b []byte
	buf := bytes.NewBuffer(b)
	for _, res := range m.Resources() {
		out, err := res.AsYAML()
		if err != nil {
			m, _ := res.Map()
			return nil, errors.Wrapf(err, "%#v", m)
		}
		if firstObj {
			firstObj = false
		} else {
			if _, err = buf.WriteString("---\n"); err != nil {
				return nil, err
			}
		}
		if _, err = buf.Write(out); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

// ErrorIfNotEqualSets implements ResMap.
func (m *resWrangler) ErrorIfNotEqualSets(other ResMap) error {
	m2, ok := other.(*resWrangler)
	if !ok {
		panic("bad cast")
	}
	if m.Size() != m2.Size() {
		return fmt.Errorf(
			"lists have different number of entries: %#v doesn't equal %#v",
			m.rList, m2.rList)
	}
	seen := make(map[int]bool)
	for _, r1 := range m.rList {
		id := r1.CurId()
		others := m2.GetMatchingResourcesByCurrentId(id.Equals)
		if len(others) == 0 {
			return fmt.Errorf(
				"id in self missing from other; id: %s", id)
		}
		if len(others) > 1 {
			return fmt.Errorf(
				"id in self matches %d in other; id: %s", len(others), id)
		}
		r2 := others[0]
		if !r1.KunstructEqual(r2) {
			return fmt.Errorf(
				"kunstruct not equal: \n -- %s,\n -- %s\n\n--\n%#v\n------\n%#v\n",
				r1, r2, r1, r2)
		}
		seen[m2.indexOfResource(r2)] = true
	}
	if len(seen) != m.Size() {
		return fmt.Errorf("counting problem %d != %d", len(seen), m.Size())
	}
	return nil
}

// ErrorIfNotEqualList implements ResMap.
func (m *resWrangler) ErrorIfNotEqualLists(other ResMap) error {
	m2, ok := other.(*resWrangler)
	if !ok {
		panic("bad cast")
	}
	if m.Size() != m2.Size() {
		return fmt.Errorf(
			"lists have different number of entries: %#v doesn't equal %#v",
			m.rList, m2.rList)
	}
	for i, r1 := range m.rList {
		r2 := m2.rList[i]
		if err := r1.ErrIfNotEquals(r2); err != nil {
			return err
		}
	}
	return nil
}

type resCopier func(r *resource.Resource) *resource.Resource

// ShallowCopy implements ResMap.
func (m *resWrangler) ShallowCopy() ResMap {
	return m.makeCopy(
		func(r *resource.Resource) *resource.Resource {
			return r
		})
}

// DeepCopy implements ResMap.
func (m *resWrangler) DeepCopy() ResMap {
	return m.makeCopy(
		func(r *resource.Resource) *resource.Resource {
			return r.DeepCopy()
		})
}

// makeCopy copies the ResMap.
func (m *resWrangler) makeCopy(copier resCopier) ResMap {
	result := &resWrangler{}
	result.rList = make([]*resource.Resource, m.Size())
	for i, r := range m.rList {
		result.rList[i] = copier(r)
	}
	return result
}

// SubsetThatCouldBeReferencedByResource implements ResMap.
func (m *resWrangler) SubsetThatCouldBeReferencedByResource(
	referrer *resource.Resource) ResMap {
	referrerId := referrer.CurId()
	if !referrerId.IsNamespaceableKind() {
		// A cluster scoped resource can refer to anything.
		return m
	}
	result := newOne()
	roleBindingNamespaces := getNamespacesForRoleBinding(referrer)
	for _, possibleTarget := range m.Resources() {
		id := possibleTarget.CurId()
		if !id.IsNamespaceableKind() {
			// A cluster-scoped resource can be referred to by anything.
			result.append(possibleTarget)
			continue
		}
		if id.IsNsEquals(referrerId) {
			// The two objects are in the same namespace.
			result.append(possibleTarget)
			continue
		}
		// The two objects are namespaced (not cluster-scoped), AND
		// are in different namespaces.
		// There's still a chance they can refer to each other.
		ns := possibleTarget.GetNamespace()
		if roleBindingNamespaces[ns] {
			result.append(possibleTarget)
		}
	}
	return result
}

// getNamespacesForRoleBinding returns referenced ServiceAccount namespaces
// if the resource is a RoleBinding
func getNamespacesForRoleBinding(r *resource.Resource) map[string]bool {
	result := make(map[string]bool)
	if r.GetKind() != "RoleBinding" {
		return result
	}
	subjects, err := r.GetSlice("subjects")
	if err != nil || subjects == nil {
		return result
	}
	for _, s := range subjects {
		subject := s.(map[string]interface{})
		if ns, ok1 := subject["namespace"]; ok1 {
			if kind, ok2 := subject["kind"]; ok2 {
				if kind.(string) == "ServiceAccount" {
					result[ns.(string)] = true
				}
			}
		}
	}
	return result
}

func (m *resWrangler) append(res *resource.Resource) {
	m.rList = append(m.rList, res)
}

// AppendAll implements ResMap.
func (m *resWrangler) AppendAll(other ResMap) error {
	if other == nil {
		return nil
	}
	for _, res := range other.Resources() {
		if err := m.Append(res); err != nil {
			return err
		}
	}
	return nil
}

// AbsorbAll implements ResMap.
func (m *resWrangler) AbsorbAll(other ResMap) error {
	if other == nil {
		return nil
	}
	for _, r := range other.Resources() {
		err := m.appendReplaceOrMerge(r)
		if err != nil {
			return err
		}
	}
	return nil
}

func (m *resWrangler) appendReplaceOrMerge(res *resource.Resource) error {
	id := res.CurId()
	matches := m.GetMatchingResourcesByAnyId(id.Equals)
	switch len(matches) {
	case 0:
		switch res.Behavior() {
		case types.BehaviorMerge, types.BehaviorReplace:
			return fmt.Errorf(
				"id %#v does not exist; cannot merge or replace", id)
		default:
			// presumably types.BehaviorCreate
			return m.Append(res)
		}
	case 1:
		old := matches[0]
		if old == nil {
			return fmt.Errorf("id lookup failure")
		}
		index := m.indexOfResource(old)
		if index < 0 {
			return fmt.Errorf("indexing problem")
		}
		switch res.Behavior() {
		case types.BehaviorReplace:
			res.CopyMergeMetaDataFieldsFrom(old)
		case types.BehaviorMerge:
			res.CopyMergeMetaDataFieldsFrom(old)
			res.MergeDataMapFrom(old)
			res.MergeBinaryDataMapFrom(old)
		default:
			return fmt.Errorf(
				"id %#v exists; behavior must be merge or replace", id)
		}
		i, err := m.Replace(res)
		if err != nil {
			return err
		}
		if i != index {
			return fmt.Errorf("unexpected target index in replacement")
		}
		return nil
	default:
		return fmt.Errorf(
			"found multiple objects %v that could accept merge of %v",
			matches, id)
	}
}

// Select returns a list of resources that
// are selected by a Selector
func (m *resWrangler) Select(s types.Selector) ([]*resource.Resource, error) {
	var result []*resource.Resource
	sr, err := types.NewSelectorRegex(&s)
	if err != nil {
		return nil, err
	}
	for _, r := range m.Resources() {
		curId := r.CurId()
		orgId := r.OrgId()

		// It first tries to match with the original namespace
		// then matches with the current namespace
		if !sr.MatchNamespace(orgId.EffectiveNamespace()) &&
			!sr.MatchNamespace(curId.EffectiveNamespace()) {
			continue
		}

		// It first tries to match with the original name
		// then matches with the current name
		if !sr.MatchName(orgId.Name) &&
			!sr.MatchName(curId.Name) {
			continue
		}

		// matches the GVK
		if !sr.MatchGvk(r.GetGvk()) {
			continue
		}

		// matches the label selector
		matched, err := r.MatchesLabelSelector(s.LabelSelector)
		if err != nil {
			return nil, err
		}
		if !matched {
			continue
		}

		// matches the annotation selector
		matched, err = r.MatchesAnnotationSelector(s.AnnotationSelector)
		if err != nil {
			return nil, err
		}
		if !matched {
			continue
		}
		result = append(result, r)
	}
	return result, nil
}

// ToRNodeSlice converts the resources in the resmp
// to a list of RNodes
func (m *resWrangler) ToRNodeSlice() ([]*kyaml_yaml.RNode, error) {
	var rnodes []*kyaml_yaml.RNode
	for _, r := range m.Resources() {
		s, err := r.AsYAML()
		if err != nil {
			return nil, err
		}
		rnode, err := kyaml_yaml.Parse(string(s))
		if err != nil {
			return nil, err
		}
		rnodes = append(rnodes, rnode)
	}
	return rnodes, nil
}

func (m *resWrangler) ApplySmPatch(
	selectedSet *resource.IdSet, patch *resource.Resource) error {
	newRm := New()
	for _, res := range m.Resources() {
		if !selectedSet.Contains(res.CurId()) {
			newRm.Append(res)
			continue
		}
		patchCopy := patch.DeepCopy()
		patchCopy.CopyMergeMetaDataFieldsFrom(patch)
		patchCopy.SetGvk(res.GetGvk())
		err := res.ApplySmPatch(patchCopy)
		if err != nil {
			// Check for an error string from UnmarshalJSON that's indicative
			// of an object that's missing basic KRM fields, and thus may have been
			// entirely deleted (an acceptable outcome).  This error handling should
			// be deleted along with use of ResMap and apimachinery functions like
			// UnmarshalJSON.
			if !strings.Contains(err.Error(), "Object 'Kind' is missing") {
				// Some unknown error, let it through.
				return err
			}
			empty, err := res.IsEmpty()
			if err != nil {
				return err
			}
			if !empty {
				m, _ := res.Map()
				return errors.Wrapf(
					err, "with unexpectedly non-empty object map of size %d",
					len(m))
			}
			// Fall through to handle deleted object.
		}
		empty, err := res.IsEmpty()
		if err != nil {
			return err
		}
		if !empty {
			// IsEmpty means all fields have been removed from the object.
			// This can happen if a patch required deletion of the
			// entire resource (not just a part of it).  This means
			// the overall resmap must shrink by one.
			newRm.Append(res)
		}
	}
	m.Clear()
	m.AppendAll(newRm)
	return nil
}

func (m *resWrangler) RemoveBuildAnnotations() {
	for _, r := range m.Resources() {
		r.RemoveBuildAnnotations()
	}
}
