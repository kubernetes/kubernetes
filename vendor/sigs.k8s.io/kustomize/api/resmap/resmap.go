// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package resmap implements a map from ResId to Resource that
// tracks all resources in a kustomization.
package resmap

import (
	"bytes"
	"fmt"
	"regexp"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/yaml"
)

// A Transformer modifies an instance of ResMap.
type Transformer interface {
	// Transform modifies data in the argument,
	// e.g. adding labels to resources that can be labelled.
	Transform(m ResMap) error
}

// A Generator creates an instance of ResMap.
type Generator interface {
	Generate() (ResMap, error)
}

// Something that's configurable accepts an
// instance of PluginHelpers and a raw config
// object (YAML in []byte form).
type Configurable interface {
	Config(h *PluginHelpers, config []byte) error
}

// NewPluginHelpers makes an instance of PluginHelpers.
func NewPluginHelpers(ldr ifc.Loader, v ifc.Validator, rf *Factory) *PluginHelpers {
	return &PluginHelpers{ldr: ldr, v: v, rf: rf}
}

// PluginHelpers holds things that any or all plugins might need.
// This should be available to each plugin, in addition to
// any plugin-specific configuration.
type PluginHelpers struct {
	ldr ifc.Loader
	v   ifc.Validator
	rf  *Factory
}

func (c *PluginHelpers) Loader() ifc.Loader {
	return c.ldr
}

func (c *PluginHelpers) ResmapFactory() *Factory {
	return c.rf
}

func (c *PluginHelpers) Validator() ifc.Validator {
	return c.v
}

type GeneratorPlugin interface {
	Generator
	Configurable
}

type TransformerPlugin interface {
	Transformer
	Configurable
}

// ResMap is an interface describing operations on the
// core kustomize data structure, a list of Resources.
//
// Every Resource has two ResIds: OrgId and CurId.
//
// In a ResMap, no two resources may have the same CurId,
// but they may have the same OrgId.  The latter can happen
// when mixing two or more different overlays apply different
// transformations to a common base.  When looking for a
// resource to transform, try the OrgId first, and if this
// fails or finds too many, it might make sense to then try
// the CurrId.  Depends on the situation.
type ResMap interface {
	// Size reports the number of resources.
	Size() int

	// Resources provides a discardable slice
	// of resource pointers, returned in the order
	// as appended.
	Resources() []*resource.Resource

	// Append adds a Resource. Error on CurId collision.
	//
	// A class invariant of ResMap is that all of its
	// resources must differ in their value of
	// CurId(), aka current Id.  The Id is the tuple
	// of {namespace, group, version, kind, name}
	// (see ResId).
	//
	// This invariant reflects the invariant of a
	// kubernetes cluster, where if one tries to add
	// a resource to the cluster whose Id matches
	// that of a resource already in the cluster,
	// only two outcomes are allowed.  Either the
	// incoming resource is _merged_ into the existing
	// one, or the incoming resource is rejected.
	// One cannot end up with two resources
	// in the cluster with the same Id.
	Append(*resource.Resource) error

	// AppendAll appends another ResMap to self,
	// failing on any OrgId collision.
	AppendAll(ResMap) error

	// AbsorbAll appends, replaces or merges the contents
	// of another ResMap into self,
	// allowing and sometimes demanding ID collisions.
	// A collision would be demanded, say, when a generated
	// ConfigMap has the "replace" option in its generation
	// instructions, meaning it _must_ replace
	// something in the known set of resources.
	// If a resource id for resource X is found to already
	// be in self, then the behavior field for X must
	// be BehaviorMerge or BehaviorReplace. If X is not in
	// self, then its behavior _cannot_ be merge or replace.
	AbsorbAll(ResMap) error

	// AsYaml returns the yaml form of resources.
	AsYaml() ([]byte, error)

	// GetByIndex returns a resource at the given index,
	// nil if out of range.
	GetByIndex(int) *resource.Resource

	// GetIndexOfCurrentId returns the index of the resource
	// with the given CurId.
	// Returns error if there is more than one match.
	// Returns (-1, nil) if there is no match.
	GetIndexOfCurrentId(id resid.ResId) (int, error)

	// GetMatchingResourcesByCurrentId returns the resources
	// who's CurId is matched by the argument.
	GetMatchingResourcesByCurrentId(matches IdMatcher) []*resource.Resource

	// GetMatchingResourcesByOriginalId returns the resources
	// who's OriginalId is matched by the argument.
	GetMatchingResourcesByOriginalId(matches IdMatcher) []*resource.Resource

	// GetByCurrentId is shorthand for calling
	// GetMatchingResourcesByCurrentId with a matcher requiring
	// an exact match, returning an error on multiple or no matches.
	GetByCurrentId(resid.ResId) (*resource.Resource, error)

	// GetByOriginalId is shorthand for calling
	// GetMatchingResourcesByOriginalId with a matcher requiring
	// an exact match, returning an error on multiple or no matches.
	GetByOriginalId(resid.ResId) (*resource.Resource, error)

	// GetById is a helper function which first
	// attempts GetByOriginalId, then GetByCurrentId,
	// returning an error if both fail to find a single
	// match.
	GetById(resid.ResId) (*resource.Resource, error)

	// GroupedByCurrentNamespace returns a map of namespace
	// to a slice of *Resource in that namespace.
	// Resources for whom IsNamespaceableKind is false are
	// are not included at all (see NonNamespaceable).
	// Resources with an empty namespace are placed
	// in the resid.DefaultNamespace entry.
	GroupedByCurrentNamespace() map[string][]*resource.Resource

	// GroupByOrginalNamespace performs as GroupByNamespace
	// but use the original namespace instead of the current
	// one to perform the grouping.
	GroupedByOriginalNamespace() map[string][]*resource.Resource

	// NonNamespaceable returns a slice of resources that
	// cannot be placed in a namespace, e.g.
	// Node, ClusterRole, Namespace itself, etc.
	NonNamespaceable() []*resource.Resource

	// AllIds returns all CurrentIds.
	AllIds() []resid.ResId

	// Replace replaces the resource with the matching CurId.
	// Error if there's no match or more than one match.
	// Returns the index where the replacement happened.
	Replace(*resource.Resource) (int, error)

	// Remove removes the resource whose CurId matches the argument.
	// Error if not found.
	Remove(resid.ResId) error

	// Clear removes all resources and Ids.
	Clear()

	// SubsetThatCouldBeReferencedByResource returns a ResMap subset
	// of self with resources that could be referenced by the
	// resource argument.
	// This is a filter; it excludes things that cannot be
	// referenced by the resource, e.g. objects in other
	// namespaces. Cluster wide objects are never excluded.
	SubsetThatCouldBeReferencedByResource(*resource.Resource) ResMap

	// DeepCopy copies the ResMap and underlying resources.
	DeepCopy() ResMap

	// ShallowCopy copies the ResMap but
	// not the underlying resources.
	ShallowCopy() ResMap

	// ErrorIfNotEqualSets returns an error if the
	// argument doesn't have the same resources as self.
	// Ordering is _not_ taken into account,
	// as this function was solely used in tests written
	// before internal resource order was maintained,
	// and those tests are initialized with maps which
	// by definition have random ordering, and will
	// fail spuriously.
	// TODO: modify tests to not use resmap.FromMap,
	// TODO: - and replace this with a stricter equals.
	ErrorIfNotEqualSets(ResMap) error

	// ErrorIfNotEqualLists returns an error if the
	// argument doesn't have the resource objects
	// data as self, in the same order.
	// Meta information is ignored; this is similar
	// to comparing the AsYaml() strings, but allows
	// for more informed errors on not equals.
	ErrorIfNotEqualLists(ResMap) error

	// Debug prints the ResMap.
	Debug(title string)

	// Select returns a list of resources that
	// are selected by a Selector
	Select(types.Selector) ([]*resource.Resource, error)
}

// resWrangler holds the content manipulated by kustomize.
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
		blob, err := yaml.Marshal(r.Map())
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

func GetOriginalId(r *resource.Resource) resid.ResId { return r.OrgId() }
func GetCurrentId(r *resource.Resource) resid.ResId  { return r.CurId() }

// GetMatchingResourcesByCurrentId implements ResMap.
func (m *resWrangler) GetMatchingResourcesByCurrentId(
	matches IdMatcher) []*resource.Resource {
	return m.filteredById(matches, GetCurrentId)
}

// GetMatchingResourcesByOriginalId implements ResMap.
func (m *resWrangler) GetMatchingResourcesByOriginalId(
	matches IdMatcher) []*resource.Resource {
	return m.filteredById(matches, GetOriginalId)
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

// GetByOriginalId implements ResMap.
func (m *resWrangler) GetByOriginalId(
	id resid.ResId) (*resource.Resource, error) {
	return demandOneMatch(m.GetMatchingResourcesByOriginalId, id, "Original")
}

// GetById implements ResMap.
func (m *resWrangler) GetById(
	id resid.ResId) (*resource.Resource, error) {
	match, err1 := m.GetByOriginalId(id)
	if err1 == nil {
		return match, nil
	}
	match, err2 := m.GetByCurrentId(id)
	if err2 == nil {
		return match, nil
	}
	return nil, fmt.Errorf(
		"%s; %s; failed to find unique target for patch %s",
		err1.Error(), err2.Error(), id.GvknString())
}

type resFinder func(IdMatcher) []*resource.Resource

func demandOneMatch(
	f resFinder, id resid.ResId, s string) (*resource.Resource, error) {
	r := f(id.Equals)
	if len(r) == 1 {
		return r[0], nil
	}
	if len(r) > 1 {
		return nil, fmt.Errorf("multiple matches for %sId %s", s, id)
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
		out, err := yaml.Marshal(res.Map())
		if err != nil {
			return nil, err
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
		if len(others) < 0 {
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
		if !r1.Equals(r2) {
			return fmt.Errorf(
				"Item i=%d differs:\n  n1 = %s\n  n2 = %s\n  o1 = %s\n  o2 = %s\n",
				i, r1.OrgId(), r2.OrgId(), r1, r2)
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
	inputRes *resource.Resource) ResMap {
	result := newOne()
	inputId := inputRes.CurId()
	isInputIdNamespaceable := inputId.IsNamespaceableKind()
	rctxm := inputRes.PrefixesSuffixesEquals
	for _, r := range m.Resources() {
		// Need to match more accuratly both at the time of selection and transformation.
		// OutmostPrefixSuffixEquals is not accurate enough since it is only using
		// the outer most suffix and the last prefix. Use PrefixedSuffixesEquals instead.
		resId := r.CurId()
		if (!isInputIdNamespaceable || !resId.IsNamespaceableKind() || resId.IsNsEquals(inputId)) &&
			r.InSameKustomizeCtx(rctxm) {
			result.append(r)
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

func (m *resWrangler) appendReplaceOrMerge(
	res *resource.Resource) error {
	id := res.CurId()
	matches := m.GetMatchingResourcesByOriginalId(id.Equals)
	if len(matches) == 0 {
		matches = m.GetMatchingResourcesByCurrentId(id.Equals)
	}
	switch len(matches) {
	case 0:
		switch res.Behavior() {
		case types.BehaviorMerge, types.BehaviorReplace:
			return fmt.Errorf(
				"id %#v does not exist; cannot merge or replace", id)
		default:
			// presumably types.BehaviorCreate
			err := m.Append(res)
			if err != nil {
				return err
			}
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
			res.Replace(old)
		case types.BehaviorMerge:
			res.Merge(old)
		default:
			return fmt.Errorf(
				"id %#v exists; must merge or replace", id)
		}
		i, err := m.Replace(res)
		if err != nil {
			return err
		}
		if i != index {
			return fmt.Errorf("unexpected index in replacement")
		}
	default:
		return fmt.Errorf(
			"found multiple objects %v that could accept merge of %v",
			matches, id)
	}
	return nil
}

func anchorRegex(pattern string) string {
	if pattern == "" {
		return pattern
	}
	return "^" + pattern + "$"
}

// Select returns a list of resources that
// are selected by a Selector
func (m *resWrangler) Select(s types.Selector) ([]*resource.Resource, error) {
	ns := regexp.MustCompile(anchorRegex(s.Namespace))
	nm := regexp.MustCompile(anchorRegex(s.Name))
	var result []*resource.Resource
	for _, r := range m.Resources() {
		curId := r.CurId()
		orgId := r.OrgId()

		// matches the namespace when namespace is not empty in the selector
		// It first tries to match with the original namespace
		// then matches with the current namespace
		if r.GetNamespace() != "" {
			matched := ns.MatchString(orgId.EffectiveNamespace())
			if !matched {
				matched = ns.MatchString(curId.EffectiveNamespace())
				if !matched {
					continue
				}
			}
		}

		// matches the name when name is not empty in the selector
		// It first tries to match with the original name
		// then matches with the current name
		if r.GetName() != "" {
			matched := nm.MatchString(orgId.Name)
			if !matched {
				matched = nm.MatchString(curId.Name)
				if !matched {
					continue
				}
			}
		}

		// matches the GVK
		if !r.GetGvk().IsSelected(&s.Gvk) {
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
