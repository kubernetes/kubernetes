/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validators

import (
	"cmp"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

var globalWrapperTags = sets.New(
	"k8s:ifEnabled",
	"k8s:ifDisabled",
	"k8s:alpha",
	"k8s:beta",
	"k8s:subfield",
)

// This is the global registry of tag validators. For simplicity this is in
// the same package as the implementations, but it should not be used directly.
var globalRegistry = &registry{
	tagValidators: map[string]TagValidator{},
}

// registry holds a list of registered tags.
type registry struct {
	lock        sync.Mutex
	initialized atomic.Bool // init() was called

	tagValidators map[string]TagValidator // keyed by tagname
	tagIndex      []string                // all tag names
}

func (reg *registry) addTagValidator(tv TagValidator) {
	if reg.initialized.Load() {
		panic("registry was modified after init")
	}

	reg.lock.Lock()
	defer reg.lock.Unlock()

	name := tv.TagName()
	if _, exists := globalRegistry.tagValidators[name]; exists {
		panic(fmt.Sprintf("tag %q was registered twice", name))
	}
	switch level := tv.Docs().StabilityLevel; level {
	case TagStabilityLevelAlpha, TagStabilityLevelBeta, TagStabilityLevelStable:
		// valid
	case "":
		panic(fmt.Sprintf("tag %q is missing stability level", name))
	default:
		panic(fmt.Sprintf("tag %q has invalid stability level %q", name, level))
	}
	globalRegistry.tagValidators[name] = tv
}

func (reg *registry) init(c *generator.Context, inputToCanonicalPkg map[string]string) {
	if reg.initialized.Load() {
		panic("registry.init() was called twice")
	}

	reg.lock.Lock()
	defer reg.lock.Unlock()

	cfg := Config{
		GengoContext:        c,
		Extractor:           reg,
		InputToCanonicalPkg: inputToCanonicalPkg,
	}

	for _, tv := range globalRegistry.tagValidators {
		reg.tagIndex = append(reg.tagIndex, tv.TagName())
		tv.Init(cfg)
	}
	sort.Strings(reg.tagIndex)

	reg.initialized.Store(true)
}

func (reg *registry) ExtractTags(_ Context, comments []string) ([]codetags.Tag, error) {
	if !reg.initialized.Load() {
		reg.init(nil, nil)
	}
	extracted := codetags.Extract("+", comments)
	var tags []codetags.Tag
	for tagName, lines := range extracted {
		if !slices.Contains(reg.tagIndex, tagName) && !globalWrapperTags.Has(tagName) {
			continue
		}
		t, err := codetags.ParseAll(lines)
		if err != nil {
			return nil, fmt.Errorf("failed to parse tags: %w: %s", err, lines)
		}
		tags = append(tags, t...)
	}
	return tags, nil
}

// ExtractValidations considers the given context (e.g. a type definition) and
// evaluates registered validators.  This includes type validators (which run
// against all types) and tag validators which run only if a specific tag is
// found in the associated comment block.  Any matching validators produce zero
// or more validations, which will later be rendered by the code-generation
// logic.
func (reg *registry) ExtractValidations(context Context, metadata SchemaMetadata, tags ...codetags.Tag) (Validations, error) {
	if !reg.initialized.Load() {
		panic("registry.init() was not called")
	}

	validations, err := reg.ExtractTagValidations(context, metadata, tags...)
	if err != nil {
		return Validations{}, err
	}
	accumulatedValidations := Validations{}
	accumulatedValidations.Add(validations)

	if context.Scope == ScopeType || context.Scope == ScopeListVal || context.Scope == ScopeMapVal {
		aggregateValidations, err := GenerateAggregateValidations(context, metadata)
		if err != nil {
			return Validations{}, err
		}
		accumulatedValidations.Add(aggregateValidations)
	}

	if context.Scope == ScopeType && len(accumulatedValidations.TypeFunctions) > 0 {
		accumulatedValidations.Functions = append(accumulatedValidations.Functions, accumulatedValidations.TypeFunctions...)
		accumulatedValidations.TypeFunctions = nil
	}
	return accumulatedValidations, nil
}

func (reg *registry) ExtractTagValidations(context Context, metadata SchemaMetadata, tags ...codetags.Tag) (Validations, error) {
	if !reg.initialized.Load() {
		panic("registry.init() was not called")
	}
	accumulatedValidations := Validations{}
	tags = reg.sortTags(tags)
	for _, tag := range tags {
		tv := reg.tagValidators[tag.Name]
		// At this point we know tv exists and is not nil due to the upfront check
		if scopes := tv.ValidScopes(); !scopes.Has(context.Scope) {
			return Validations{}, fmt.Errorf("tag %q cannot be specified on %s", tv.TagName(), context.Scope)
		}
		if err := typeCheck(tag, tv.Docs()); err != nil {
			return Validations{}, fmt.Errorf("tag %q: %w", tv.TagName(), err)
		}
		if emitter, ok := tv.(ValidationEmitter); ok {
			group, err := emitter.GetValidations(context, metadata, tag)
			if err != nil {
				return Validations{}, fmt.Errorf("tag %q: %w", tv.TagName(), err)
			}
			theseValidations, err := FinalizeGroup(context, group)
			if err != nil {
				return Validations{}, fmt.Errorf("tag %q finalizer: %w", tv.TagName(), err)
			}
			accumulatedValidations.Add(theseValidations)
		}
	}
	return accumulatedValidations, nil
}

func (reg *registry) sortTags(tags []codetags.Tag) []codetags.Tag {
	// First sort all tags by their name, so the final output is deterministic.
	// It is important to do this before validations are generated.
	//
	// Some tags are "meta" tags which wrap other tags. For example:
	//
	//   // +k8s:validateFalse="111"
	//   // +k8s:validateFalse="222"
	//   // +k8s:ifEnabled(Foo)=+k8s:validateFalse="333"
	//
	// Tag extraction will group these by tag name. The first two are
	// instances of "k8s:validateFalse", while the third is an instance of
	// "k8s:ifEnabled".
	//
	// Without sorting, the order in which tag validators are called is not defined
	// (map iteration). This can lead to non-deterministic order of the generated
	// validations. By sorting the tags by name first, we ensure that "k8s:ifEnabled"
	// is processed before or after "k8s:validateFalse" consistently, allowing the
	// "k8s:validateFalse" tags to remain grouped together. The tags for each name
	// are processed in order of appearance, so relative ordering is preserved.
	sortedTags := make([]codetags.Tag, len(tags))
	copy(sortedTags, tags)

	slices.SortFunc(sortedTags, func(a, b codetags.Tag) int {
		return cmp.Compare(a.Name, b.Name)
	})

	return sortedTags
}

// Docs returns documentation for each tag in this registry.
func (reg *registry) Docs() []TagDoc {
	var result []TagDoc
	for _, k := range reg.tagIndex {
		v := reg.tagValidators[k]
		result = append(result, v.Docs())
	}
	return result
}

func RegisterTagValidator(tv TagValidator) {
	globalRegistry.addTagValidator(tv)
}

// Conditional wraps any metadata constraint payload with the conditions and stability level
// under which it is active. This ensures uniform conditional modeling across all structural
// and validation rules.
type Conditional[T any] struct {
	Conditions     Conditions
	StabilityLevel ValidationStabilityLevel
	Payload        T
}

// NodeMetadata holds all schema and validation constraints scoped to a specific canonical AST path.
// Opaque is the only unconditioned metadata; all other constraints can be conditionally gated (+k8s:ifEnabled).
type NodeMetadata struct {
	Path   NodePath
	Opaque bool

	Lists             []Conditional[*listMetadata]
	UpdateConstraints []Conditional[*updateMetadata]
	Dependencies      []Conditional[*dependencyMetadata]
	Unions            []Conditional[*union]
	Modes             []Conditional[*discriminatorGroup]
	Items             []Conditional[*itemMetadata]
}

// Merge combines constraints from other into nm, appending conditional constraint slices.
func (nm *NodeMetadata) Merge(other *NodeMetadata) error {
	if other == nil {
		return nil
	}
	nm.Opaque = nm.Opaque || other.Opaque
	var err error
	nm.Lists, err = mergeListConditionals(nm.Lists, other.Lists)
	if err != nil {
		return err
	}
	nm.UpdateConstraints = mergeUpdateConstraints(nm.UpdateConstraints, other.UpdateConstraints)
	nm.Dependencies = append(nm.Dependencies, other.Dependencies...)
	nm.Unions, err = mergeUnionConditionals(nm.Unions, other.Unions)
	if err != nil {
		return err
	}
	nm.Modes, err = mergeModeConditionals(nm.Modes, other.Modes)
	if err != nil {
		return err
	}
	nm.Items = append(nm.Items, other.Items...)
	return nil
}

func sameParentPath(p1, p2 *field.Path) bool {
	if p1 == nil && p2 == nil {
		return true
	}
	if p1 == nil || p2 == nil {
		return false
	}
	return p1.String() == p2.String()
}

func mergeListConditionals(existing, other []Conditional[*listMetadata]) ([]Conditional[*listMetadata], error) {
	res := existing
	for _, o := range other {
		merged := false
		for i, e := range res {
			if e.Payload != nil && o.Payload != nil && sameParentPath(e.Payload.targetPath, o.Payload.targetPath) {
				if err := res[i].Payload.merge(o.Payload); err != nil {
					return nil, err
				}
				if res[i].StabilityLevel == "" && o.StabilityLevel != "" {
					res[i].StabilityLevel = o.StabilityLevel
				}
				res[i].Conditions = res[i].Conditions.Merge(o.Conditions)
				merged = true
				break
			}
		}
		if !merged {
			res = append(res, o)
		}
	}
	return res, nil
}

func mergeUnionConditionals(existing, other []Conditional[*union]) ([]Conditional[*union], error) {
	res := existing
	for _, o := range other {
		merged := false
		for i, e := range res {
			if e.Payload != nil && o.Payload != nil &&
				e.Payload.name == o.Payload.name &&
				sameParentPath(e.Payload.parentPath, o.Payload.parentPath) &&
				e.Conditions.Key() == o.Conditions.Key() &&
				e.StabilityLevel == o.StabilityLevel {
				if err := res[i].Payload.Merge(o.Payload); err != nil {
					return nil, err
				}
				merged = true
				break
			}
		}
		if !merged {
			res = append(res, o)
		}
	}
	return res, nil
}

func mergeModeConditionals(existing, other []Conditional[*discriminatorGroup]) ([]Conditional[*discriminatorGroup], error) {
	res := existing
	for _, o := range other {
		merged := false
		for i, e := range res {
			if e.Payload != nil && o.Payload != nil && e.Payload.name == o.Payload.name &&
				e.Conditions.Key() == o.Conditions.Key() &&
				e.StabilityLevel == o.StabilityLevel {
				if err := res[i].Payload.merge(o.Payload); err != nil {
					return nil, err
				}
				merged = true
				break
			}
		}
		if !merged {
			res = append(res, o)
		}
	}
	return res, nil
}

func mergeUpdateConstraints(existing, other []Conditional[*updateMetadata]) []Conditional[*updateMetadata] {
	res := existing
	for _, o := range other {
		merged := false
		for i, e := range res {
			if e.Payload != nil && o.Payload != nil && e.Conditions.Key() == o.Conditions.Key() {
				res[i].Payload.merge(o.Payload)
				if res[i].StabilityLevel == "" && o.StabilityLevel != "" {
					res[i].StabilityLevel = o.StabilityLevel
				}
				merged = true
				break
			}
		}
		if !merged {
			res = append(res, o)
		}
	}
	return res
}

// IsEmpty returns true if nm has no validation constraints or flags set.
func (nm *NodeMetadata) IsEmpty() bool {
	if nm == nil {
		return true
	}
	return !nm.Opaque &&
		len(nm.Lists) == 0 &&
		len(nm.UpdateConstraints) == 0 &&
		len(nm.Dependencies) == 0 &&
		len(nm.Unions) == 0 &&
		len(nm.Items) == 0 &&
		len(nm.Modes) == 0
}

// SchemaMetadata represents the consolidated metadata tree for a Go type or field,
// indexed by canonical AST relative path (e.g. "" for root field/typedef, "spec" for field, "spec.template" for subfield).
type SchemaMetadata struct {
	Nodes map[NodePath]*NodeMetadata
	// Track whether a specific structural generator has already emitted its type-level function
	emitted map[string]bool
}

// MarkEmitted checks if the given key has been emitted, and marks it if not.
// Returns true if it was ALREADY emitted, false if it is the first time.
func (sm *SchemaMetadata) MarkEmitted(key string) bool {
	if sm.emitted == nil {
		sm.emitted = make(map[string]bool)
	}
	if sm.emitted[key] {
		return true
	}
	sm.emitted[key] = true
	return false
}

// NodePath is the canonical key for SchemaMetadata.Nodes: the string form of
// an absolute context path. Derive values with nodeKeyFor rather than from raw
// strings, so that producers and consumers always agree and lookups are exact.
type NodePath string

// nodeKeyFor returns the canonical Nodes key for an absolute context path.
func nodeKeyFor(p *field.Path) NodePath {
	if p == nil {
		return ""
	}
	return NodePath(p.String())
}

// GetOrCreateNode retrieves the NodeMetadata for path, creating it if it does not exist.
func (sm *SchemaMetadata) GetOrCreateNode(path NodePath) *NodeMetadata {
	if sm.Nodes == nil {
		sm.Nodes = make(map[NodePath]*NodeMetadata)
	}
	node, ok := sm.Nodes[path]
	if !ok {
		node = &NodeMetadata{Path: path}
		sm.Nodes[path] = node
	}
	return node
}

// SortedNodes returns all NodeMetadata items sorted alphabetically by Path to guarantee deterministic output.
func (sm *SchemaMetadata) SortedNodes() []*NodeMetadata {
	if len(sm.Nodes) == 0 {
		return nil
	}
	paths := make([]NodePath, 0, len(sm.Nodes))
	for p := range sm.Nodes {
		paths = append(paths, p)
	}
	slices.Sort(paths)
	res := make([]*NodeMetadata, 0, len(paths))
	for _, p := range paths {
		res = append(res, sm.Nodes[p])
	}
	return res
}

// Merge merges all nodes from other into sm, matching by canonical AST path.
func (sm *SchemaMetadata) Merge(other SchemaMetadata) error {
	for path, otherNode := range other.Nodes {
		node := sm.GetOrCreateNode(path)
		if err := node.Merge(otherNode); err != nil {
			return err
		}
	}
	if len(other.emitted) > 0 {
		if sm.emitted == nil {
			sm.emitted = make(map[string]bool)
		}
		for k, v := range other.emitted {
			sm.emitted[k] = v
		}
	}
	return nil
}

func (sm SchemaMetadata) SortedUnions() []*union {
	var res []*union
	for _, node := range sm.SortedNodes() {
		for _, uCond := range node.Unions {
			u := uCond.Payload.DeepCopy()
			if !uCond.Conditions.Empty() {
				u.conditions = u.conditions.Merge(uCond.Conditions)
			}
			if uCond.StabilityLevel != "" {
				u.stabilityLevel = uCond.StabilityLevel
			}
			res = append(res, u)
		}
	}
	return res
}

func (sm SchemaMetadata) Modes() (discriminatorGroups, error) {
	res := make(discriminatorGroups)
	for _, node := range sm.SortedNodes() {
		for _, mCond := range node.Modes {
			if existing, ok := res[mCond.Payload.name]; ok {
				if err := existing.merge(mCond.Payload); err != nil {
					return nil, err
				}
			} else {
				res[mCond.Payload.name] = mCond.Payload.DeepCopy()
			}
		}
	}
	if len(res) == 0 {
		return nil, nil
	}
	return res, nil
}

func (sm SchemaMetadata) SortedDependencies() []dependencyMetadata {
	var res []dependencyMetadata
	for _, node := range sm.SortedNodes() {
		for _, dCond := range node.Dependencies {
			d := dCond.Payload.DeepCopy()
			if !dCond.Conditions.Empty() {
				d.conditions = d.conditions.Merge(dCond.Conditions)
			}
			if dCond.StabilityLevel != "" {
				d.stabilityLevel = dCond.StabilityLevel
			}
			res = append(res, d)
		}
	}
	if len(res) == 0 {
		return nil
	}
	slices.SortFunc(res, func(a, b dependencyMetadata) int {
		return a.Compare(b)
	})
	return res
}

func (sm SchemaMetadata) SortedUpdateConstraints() []Conditional[*updateMetadata] {
	var res []Conditional[*updateMetadata]
	for _, node := range sm.SortedNodes() {
		res = append(res, node.UpdateConstraints...)
	}
	return res
}

// MetadataCollector allows a TagValidator to contribute path-indexed schema and validation
// metadata to the consolidated SchemaMetadata tree.
type MetadataCollector interface {
	CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error)
}

// ValidationExtractor represents an aggregation of validator plugins.
type ValidationExtractor interface {
	Extractor

	// Docs returns documentation for each known tag.
	Docs() []TagDoc

	// Stability returns the stability level for a given tag.
	Stability(tag string) (TagStabilityLevel, error)

	// IsKnownTag returns true if the tag is a registered validation tag.
	IsKnownTag(tag string) bool

	// ExtractMetadata extracts path-indexed schema and validation metadata associated with the given tags.
	ExtractMetadata(context Context, tags ...codetags.Tag) (SchemaMetadata, error)

	// CollectMetadata collects path-indexed schema and validation metadata recursively for a given context.
	CollectMetadata(context Context) (SchemaMetadata, error)
}

func (reg *registry) ExtractMetadata(context Context, tags ...codetags.Tag) (SchemaMetadata, error) {
	if !reg.initialized.Load() {
		reg.init(nil, nil)
	}
	result := SchemaMetadata{
		emitted: make(map[string]bool),
	}
	for _, tag := range tags {
		tv, ok := reg.tagValidators[tag.Name]
		if !ok {
			continue
		}
		if collector, ok := tv.(MetadataCollector); ok {
			meta, err := collector.CollectMetadata(context, tag)
			if err != nil {
				return SchemaMetadata{}, err
			}
			if err := result.Merge(meta); err != nil {
				return SchemaMetadata{}, err
			}
		}
	}
	return result, nil
}

func (reg *registry) CollectMetadata(context Context) (SchemaMetadata, error) {
	if !reg.initialized.Load() {
		reg.init(nil, nil)
	}
	result := SchemaMetadata{
		emitted: make(map[string]bool),
	}

	if context.Member != nil && len(context.Member.CommentLines) > 0 {
		tags, err := reg.ExtractTags(context, context.Member.CommentLines)
		if err != nil {
			return SchemaMetadata{}, err
		}
		meta, err := reg.ExtractMetadata(context, tags...)
		if err != nil {
			return SchemaMetadata{}, err
		}
		if err := result.Merge(meta); err != nil {
			return SchemaMetadata{}, err
		}
	}

	if context.Type != nil && len(context.Type.CommentLines) > 0 {
		tags, err := reg.ExtractTags(context, context.Type.CommentLines)
		if err != nil {
			return SchemaMetadata{}, err
		}
		meta, err := reg.ExtractMetadata(context, tags...)
		if err != nil {
			return SchemaMetadata{}, err
		}
		if err := result.Merge(meta); err != nil {
			return SchemaMetadata{}, err
		}
	}

	if context.Type != nil {
		st := util.NonPointer(util.NativeType(context.Type))
		if st.Kind == types.Struct {
			structPath := context.Path
			for i := range st.Members {
				member := &st.Members[i]
				if len(member.CommentLines) == 0 {
					continue
				}
				childPath := structPath.Child(member.Name)
				if jsonAnnotation, ok := tags.LookupJSON(*member); ok && jsonAnnotation.Name != "" {
					childPath = structPath.Child(jsonAnnotation.Name)
				}
				fieldContext := Context{
					Scope:          ScopeField,
					Type:           member.Type,
					Member:         member,
					Path:           childPath,
					ParentPath:     structPath,
					ParentType:     context.Type,
					StabilityLevel: context.StabilityLevel,
					Conditions:     context.Conditions,
				}
				mTags, err := reg.ExtractTags(fieldContext, member.CommentLines)
				if err != nil {
					return SchemaMetadata{}, err
				}
				meta, err := reg.ExtractMetadata(fieldContext, mTags...)
				if err != nil {
					return SchemaMetadata{}, err
				}
				if err := result.Merge(meta); err != nil {
					return SchemaMetadata{}, err
				}

				currType := member.Type
				for currType != nil && currType.Name.Name != "" {
					if len(currType.CommentLines) > 0 {
						tTags, err := reg.ExtractTags(fieldContext, currType.CommentLines)
						if err != nil {
							return SchemaMetadata{}, err
						}
						tMeta, err := reg.ExtractMetadata(fieldContext, tTags...)
						if err != nil {
							return SchemaMetadata{}, err
						}
						if tNode, ok := tMeta.Nodes[nodeKeyFor(childPath)]; ok {
							// Inherit list structural metadata (e.g., listType, listMapKey, unique) from
							// the underlying typedef alias if not overridden on the field itself. We do
							// not inherit element validation rules (such as +k8s:item tags) here because
							// those are executed by the typedef's own generated validation function.
							node := result.GetOrCreateNode(nodeKeyFor(childPath))
							if len(node.Lists) == 0 && len(tNode.Lists) > 0 {
								node.Lists = tNode.Lists
							}
						}
					}
					if currType.Kind != types.Alias {
						break
					}
					currType = currType.Underlying
				}
			}
		}
	}
	return result, nil
}

// Stability returns the stability level for a given tag.
func (reg *registry) Stability(tag string) (TagStabilityLevel, error) {
	tagName := strings.TrimPrefix(tag, "+")
	tv, ok := reg.tagValidators[tagName]
	if !ok {
		return "", fmt.Errorf("tag %q doesn't have stability level", tag)
	}

	if tv.Docs().StabilityLevel == "" {
		return "", fmt.Errorf("tag %q doesn't have stability level", tag)
	}
	return tv.Docs().StabilityLevel, nil
}

// GetStability returns the stability level for a given tag from the global registry.
func GetStability(tag string) (TagStabilityLevel, error) {
	return globalRegistry.Stability(tag)
}

// IsKnownTag returns true if the tag has been registered as a validation tag.
func (reg *registry) IsKnownTag(tag string) bool {
	_, err := reg.Stability(tag)
	return err == nil
}

// IsKnownTag returns true if the given tag is a registered validation tag.
func IsKnownTag(tag string) bool {
	return globalRegistry.IsKnownTag(tag)
}

// InitGlobalValidator must be called exactly once by the main application to
// initialize and safely access the global tag registry.  Once this is called,
// no more validators may be registered.
func InitGlobalValidator(c *generator.Context, inputToCanonicalPkg map[string]string) ValidationExtractor {
	globalRegistry.init(c, inputToCanonicalPkg)
	return globalRegistry
}
