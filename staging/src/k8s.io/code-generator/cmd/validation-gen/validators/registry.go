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
	"sync"
	"sync/atomic"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
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

	typeValidators []TypeValidator
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
	globalRegistry.tagValidators[name] = tv
}

func (reg *registry) addTypeValidator(tv TypeValidator) {
	if reg.initialized.Load() {
		panic("registry was modified after init")
	}

	reg.lock.Lock()
	defer reg.lock.Unlock()

	globalRegistry.typeValidators = append(globalRegistry.typeValidators, tv)
}

func (reg *registry) init(c *generator.Context) {
	if reg.initialized.Load() {
		panic("registry.init() was called twice")
	}

	reg.lock.Lock()
	defer reg.lock.Unlock()

	cfg := Config{
		GengoContext: c,
		Validator:    reg,
	}

	for _, tv := range globalRegistry.tagValidators {
		reg.tagIndex = append(reg.tagIndex, tv.TagName())
		tv.Init(cfg)
	}
	sort.Strings(reg.tagIndex)

	for _, tv := range reg.typeValidators {
		tv.Init(cfg)
	}
	slices.SortFunc(reg.typeValidators, func(a, b TypeValidator) int {
		return cmp.Compare(a.Name(), b.Name())
	})

	reg.initialized.Store(true)
}

// ExtractValidations considers the given context (e.g. a type definition) and
// evaluates registered validators.  This includes type validators (which run
// against all types) and tag validators which run only if a specific tag is
// found in the associated comment block.  Any matching validators produce zero
// or more validations, which will later be rendered by the code-generation
// logic.
func (reg *registry) ExtractValidations(context Context, comments []string) (Validations, error) {
	if !reg.initialized.Load() {
		panic("registry.init() was not called")
	}

	validations := Validations{}

	// Extract tags and run matching tag-validators first.
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", reg.tagIndex, comments)
	if err != nil {
		return Validations{}, fmt.Errorf("failed to parse tags: %w", err)
	}
	phases := reg.sortTagsIntoPhases(tags)
	for _, idx := range phases {
		for _, tag := range idx {
			vals := tags[tag]
			tv := reg.tagValidators[tag]
			if scopes := tv.ValidScopes(); !scopes.Has(context.Scope) && !scopes.Has(ScopeAny) {
				return Validations{}, fmt.Errorf("tag %q cannot be specified on %s", tv.TagName(), context.Scope)
			}
			for _, val := range vals { // tags may have multiple values
				if theseValidations, err := tv.GetValidations(context, val.Args, val.Value); err != nil {
					return Validations{}, fmt.Errorf("tag %q: %w", tv.TagName(), err)
				} else {
					validations.Add(theseValidations)
				}
			}
		}
	}

	// Run type-validators after tag validators are done.
	if context.Scope == ScopeType {
		// Run all type-validators.
		for _, tv := range reg.typeValidators {
			if theseValidations, err := tv.GetValidations(context); err != nil {
				return Validations{}, fmt.Errorf("type validator %q: %w", tv.Name(), err)
			} else {
				validations.Add(theseValidations)
			}
		}
	}

	return validations, nil
}

func (reg *registry) sortTagsIntoPhases(tags map[string][]gengo.Tag) [][]string {
	// First sort all tags by their name, so the final output is deterministic.
	//
	// It makes more sense to sort here, rather than when emitting because:
	//
	// Consider a type or field with the following comments:
	//
	//    // +k8s:validateFalse="111"
	//    // +k8s:validateFalse="222"
	//    // +k8s:ifOptionEnabled(Foo)=+k8s:validateFalse="333"
	//
	// Tag extraction will retain the relative order between 111 and 222, but
	// 333 is extracted as tag "k8s:ifOptionEnabled".  Those are all in a map,
	// which we iterate (in a random order).  When it reaches the emit stage,
	// the "ifOptionEnabled" part is gone, and we will have 3 functionGen
	// objects, all with tag "k8s:validateFalse", in a non-deterministic order
	// because of the map iteration.  If we sort them at that point, we won't
	// have enough information to do something smart, unless we look at the
	// args, which are opaque to us.
	//
	// Sorting it earlier means we can sort "k8s:ifOptionEnabled" against
	// "k8s:validateFalse".  All of the records within each of those is
	// relatively ordered, so the result here would be to put "ifOptionEnabled"
	// before "validateFalse" (lexicographical is better than random).
	sortedTags := []string{}
	for tag := range tags {
		sortedTags = append(sortedTags, tag)
	}
	sort.Strings(sortedTags)

	// Now split them into phases.
	phase0 := []string{} // regular tags
	phase1 := []string{} // "late" tags
	for _, tn := range sortedTags {
		tv := reg.tagValidators[tn]
		if _, ok := tv.(LateTagValidator); ok {
			phase1 = append(phase1, tn)
		} else {
			phase0 = append(phase0, tn)
		}
	}
	return [][]string{phase0, phase1}
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

// RegisterTagValidator must be called by any validator which wants to run when
// a specific tag is found.
func RegisterTagValidator(tv TagValidator) {
	globalRegistry.addTagValidator(tv)
}

// RegisterTypeValidator must be called by any validator which wants to run
// against every type definition.
func RegisterTypeValidator(tv TypeValidator) {
	globalRegistry.addTypeValidator(tv)
}

// Validator represents an aggregation of validator plugins.
type Validator interface {
	// ExtractValidations considers the given context (e.g. a type definition) and
	// evaluates registered validators.  This includes type validators (which run
	// against all types) and tag validators which run only if a specific tag is
	// found in the associated comment block.  Any matching validators produce zero
	// or more validations, which will later be rendered by the code-generation
	// logic.
	ExtractValidations(context Context, comments []string) (Validations, error)

	// Docs returns documentation for each known tag.
	Docs() []TagDoc
}

// InitGlobalValidator must be called exactly once by the main application to
// initialize and safely access the global tag registry.  Once this is called,
// no more validators may be registered.
func InitGlobalValidator(c *generator.Context) Validator {
	globalRegistry.init(c)
	return globalRegistry
}
