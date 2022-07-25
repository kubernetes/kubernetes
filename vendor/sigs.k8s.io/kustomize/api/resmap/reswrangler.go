// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmap

import (
	"bytes"
	"fmt"
	"reflect"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/filters/annotations"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/resid"
	kyaml "sigs.k8s.io/kustomize/kyaml/yaml"
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

// DropEmpties quickly drops empty resources.
// It doesn't use Append, which checks for Id collisions.
func (m *resWrangler) DropEmpties() {
	var rList []*resource.Resource
	for _, r := range m.rList {
		if !r.IsNilOrEmpty() {
			rList = append(rList, r)
		}
	}
	m.rList = rList
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
	m.append(res)
	return nil
}

// append appends without performing an Id check
func (m *resWrangler) append(res *resource.Resource) {
	m.rList = append(m.rList, res)
}

// Remove implements ResMap.
func (m *resWrangler) Remove(adios resid.ResId) error {
	var rList []*resource.Resource
	for _, r := range m.rList {
		if r.CurId() != adios {
			rList = append(rList, r)
		}
	}
	if len(rList) != m.Size()-1 {
		return fmt.Errorf("id %s not found in removal", adios)
	}
	m.rList = rList
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
		fmt.Printf("# %d  %s\n%s\n", i, r.OrgId(), r.String())
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
			err.Error(), id.String())
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
	return nil, fmt.Errorf("no matches for %s %s", s, id)
}

// GroupedByCurrentNamespace implements ResMap.
func (m *resWrangler) GroupedByCurrentNamespace() map[string][]*resource.Resource {
	items := m.groupedByCurrentNamespace()
	delete(items, resid.TotallyNotANamespace)
	return items
}

// ClusterScoped implements ResMap.
func (m *resWrangler) ClusterScoped() []*resource.Resource {
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

// GroupedByOriginalNamespace implements ResMap.
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
	for _, res := range m.rList {
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
		return fmt.Errorf("bad cast to resWrangler 1")
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
		if !reflect.DeepEqual(r1.RNode, r2.RNode) {
			return fmt.Errorf(
				"nodes unequal: \n -- %s,\n -- %s\n\n--\n%#v\n------\n%#v\n",
				r1, r2, r1, r2)
		}
		seen[m2.indexOfResource(r2)] = true
	}
	if len(seen) != m.Size() {
		return fmt.Errorf("counting problem %d != %d", len(seen), m.Size())
	}
	return nil
}

// ErrorIfNotEqualLists implements ResMap.
func (m *resWrangler) ErrorIfNotEqualLists(other ResMap) error {
	m2, ok := other.(*resWrangler)
	if !ok {
		return fmt.Errorf("bad cast to resWrangler 2")
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
	referrer *resource.Resource) (ResMap, error) {
	referrerId := referrer.CurId()
	if referrerId.IsClusterScoped() {
		// A cluster scoped resource can refer to anything.
		return m, nil
	}
	result := newOne()
	roleBindingNamespaces, err := getNamespacesForRoleBinding(referrer)
	if err != nil {
		return nil, err
	}
	for _, possibleTarget := range m.rList {
		id := possibleTarget.CurId()
		if id.IsClusterScoped() {
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
		if roleBindingNamespaces[possibleTarget.GetNamespace()] {
			result.append(possibleTarget)
		}
	}
	return result, nil
}

// getNamespacesForRoleBinding returns referenced ServiceAccount namespaces
// if the resource is a RoleBinding
func getNamespacesForRoleBinding(r *resource.Resource) (map[string]bool, error) {
	result := make(map[string]bool)
	if r.GetKind() != "RoleBinding" {
		return result, nil
	}
	//nolint staticcheck
	subjects, err := r.GetSlice("subjects")
	if err != nil || subjects == nil {
		return result, nil
	}
	for _, s := range subjects {
		subject := s.(map[string]interface{})
		if ns, ok1 := subject["namespace"]; ok1 {
			if kind, ok2 := subject["kind"]; ok2 {
				if kind.(string) == "ServiceAccount" {
					if n, ok3 := ns.(string); ok3 {
						result[n] = true
					} else {
						return nil, errors.New(fmt.Sprintf("Invalid Input: namespace is blank for resource %q\n", r.CurId()))
					}
				}
			}
		}
	}
	return result, nil
}

// AppendAll implements ResMap.
func (m *resWrangler) AppendAll(other ResMap) error {
	if other == nil {
		return nil
	}
	m2, ok := other.(*resWrangler)
	if !ok {
		return fmt.Errorf("bad cast to resWrangler 3")
	}
	return m.appendAll(m2.rList)
}

// appendAll appends all the resources, error on Id collision.
func (m *resWrangler) appendAll(list []*resource.Resource) error {
	for _, res := range list {
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
	m2, ok := other.(*resWrangler)
	if !ok {
		return fmt.Errorf("bad cast to resWrangler 4")
	}
	for _, r := range m2.rList {
		err := m.appendReplaceOrMerge(r)
		if err != nil {
			return err
		}
	}
	return nil
}

// AddOriginAnnotation implements ResMap.
func (m *resWrangler) AddOriginAnnotation(origin *resource.Origin) error {
	if origin == nil {
		return nil
	}
	for _, res := range m.rList {
		or, err := res.GetOrigin()
		if or != nil || err != nil {
			// if any resources already have an origin annotation,
			// skip it
			continue
		}
		if err := res.SetOrigin(origin); err != nil {
			return err
		}
	}
	return nil
}

// RemoveOriginAnnotation implements ResMap
func (m *resWrangler) RemoveOriginAnnotations() error {
	for _, res := range m.rList {
		if err := res.SetOrigin(nil); err != nil {
			return err
		}
	}
	return nil
}

// AddTransformerAnnotation implements ResMap
func (m *resWrangler) AddTransformerAnnotation(origin *resource.Origin) error {
	for _, res := range m.rList {
		or, err := res.GetOrigin()
		if err != nil {
			return err
		}
		if or == nil {
			// the resource does not have an origin annotation, so
			// we assume that the transformer generated the resource
			// rather than modifying it
			err = res.SetOrigin(origin)
		} else {
			// the resource already has an origin annotation, so we
			// record the provided origin as a transformation
			err = res.AddTransformation(origin)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// RemoveTransformerAnnotations implements ResMap
func (m *resWrangler) RemoveTransformerAnnotations() error {
	for _, res := range m.rList {
		if err := res.ClearTransformations(); err != nil {
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
			// ensure the origin annotation doesn't get overwritten
			orig, err := old.GetOrigin()
			if err != nil {
				return err
			}
			res.CopyMergeMetaDataFieldsFrom(old)
			res.MergeDataMapFrom(old)
			res.MergeBinaryDataMapFrom(old)
			if orig != nil {
				res.SetOrigin(orig)
			}

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

// AnnotateAll implements ResMap
func (m *resWrangler) AnnotateAll(key string, value string) error {
	return m.ApplyFilter(annotations.Filter{
		Annotations: map[string]string{
			key: value,
		},
		FsSlice: []types.FieldSpec{{
			Path:               "metadata/annotations",
			CreateIfNotPresent: true,
		}},
	})
}

// Select returns a list of resources that
// are selected by a Selector
func (m *resWrangler) Select(s types.Selector) ([]*resource.Resource, error) {
	var result []*resource.Resource
	sr, err := types.NewSelectorRegex(&s)
	if err != nil {
		return nil, err
	}
	for _, r := range m.rList {
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

// ToRNodeSlice returns a copy of the resources as RNodes.
func (m *resWrangler) ToRNodeSlice() []*kyaml.RNode {
	result := make([]*kyaml.RNode, len(m.rList))
	for i := range m.rList {
		result[i] = m.rList[i].Copy()
	}
	return result
}

// DeAnchor implements ResMap.
func (m *resWrangler) DeAnchor() (err error) {
	for i := range m.rList {
		if err = m.rList[i].DeAnchor(); err != nil {
			return err
		}
	}
	return nil
}

// ApplySmPatch applies the patch, and errors on Id collisions.
func (m *resWrangler) ApplySmPatch(
	selectedSet *resource.IdSet, patch *resource.Resource) error {
	var list []*resource.Resource
	for _, res := range m.rList {
		if selectedSet.Contains(res.CurId()) {
			patchCopy := patch.DeepCopy()
			patchCopy.CopyMergeMetaDataFieldsFrom(patch)
			patchCopy.SetGvk(res.GetGvk())
			patchCopy.SetKind(patch.GetKind())
			if err := res.ApplySmPatch(patchCopy); err != nil {
				return err
			}
		}
		if !res.IsNilOrEmpty() {
			list = append(list, res)
		}
	}
	m.Clear()
	return m.appendAll(list)
}

func (m *resWrangler) RemoveBuildAnnotations() {
	for _, r := range m.rList {
		r.RemoveBuildAnnotations()
	}
}

// ApplyFilter implements ResMap.
func (m *resWrangler) ApplyFilter(f kio.Filter) error {
	reverseLookup := make(map[*kyaml.RNode]*resource.Resource, len(m.rList))
	nodes := make([]*kyaml.RNode, len(m.rList))
	for i, r := range m.rList {
		ptr := &(r.RNode)
		nodes[i] = ptr
		reverseLookup[ptr] = r
	}
	// The filter can modify nodes, but also delete and create them.
	// The filtered list might be smaller or larger than the nodes list.
	filtered, err := f.Filter(nodes)
	if err != nil {
		return err
	}
	// Rebuild the resmap from the filtered RNodes.
	var nRList []*resource.Resource
	for _, rn := range filtered {
		if rn.IsNilOrEmpty() {
			// A node might make it through the filter as an object,
			// but still be empty.  Drop such entries.
			continue
		}
		res, ok := reverseLookup[rn]
		if !ok {
			// A node was created; make a Resource to wrap it.
			res = &resource.Resource{
				RNode: *rn,
				// Leave remaining fields empty.
				// At at time of writing, seeking to eliminate those fields.
				// Alternatively, could just return error on creation attempt
				// until remaining fields eliminated.
			}
		}
		nRList = append(nRList, res)
	}
	m.rList = nRList
	return nil
}
