/*
Copyright 2018 The Kubernetes Authors.

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

package fixture

import (
	"bytes"
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// For the sake of tests, a parser is something that can retrieve a
// ParseableType.
type Parser interface {
	Type(string) typed.ParseableType
}

// SameVersionParser can be used if all the versions are actually using the same type.
type SameVersionParser struct {
	T typed.ParseableType
}

func (p SameVersionParser) Type(_ string) typed.ParseableType {
	return p.T
}

// DeducedParser is a parser that is deduced no matter what the version
// specified.
var DeducedParser = SameVersionParser{
	T: typed.DeducedParseableType,
}

// State of the current test in terms of live object. One can check at
// any time that Live and Managers match the expectations.
//
// The parser will look for the type by using the APIVersion of the
// object it's trying to parse. If trying to parse a "v1" object, a
// corresponding "v1" type should exist in the schema. If all the
// versions should map to the same type, or to a DeducedParseableType,
// one can use the SameVersionParser or the DeducedParser types defined
// in this package.
type State struct {
	Live     *typed.TypedValue
	Parser   Parser
	Managers fieldpath.ManagedFields
	Updater  *merge.Updater
}

// FixTabsOrDie counts the number of tab characters preceding the first
// line in the given yaml object. It removes that many tabs from every
// line. It panics (it's a test funtion) if some line has fewer tabs
// than the first line.
//
// The purpose of this is to make it easier to read tests.
func FixTabsOrDie(in typed.YAMLObject) typed.YAMLObject {
	lines := bytes.Split([]byte(in), []byte{'\n'})
	if len(lines[0]) == 0 && len(lines) > 1 {
		lines = lines[1:]
	}
	// Create prefix made of tabs that we want to remove.
	var prefix []byte
	for _, c := range lines[0] {
		if c != '\t' {
			break
		}
		prefix = append(prefix, byte('\t'))
	}
	// Remove prefix from all tabs, fail otherwise.
	for i := range lines {
		line := lines[i]
		// It's OK for the last line to be blank (trailing \n)
		if i == len(lines)-1 && len(line) <= len(prefix) && bytes.TrimSpace(line) == nil {
			lines[i] = []byte{}
			break
		}
		if !bytes.HasPrefix(line, prefix) {
			panic(fmt.Errorf("line %d doesn't start with expected number (%d) of tabs: %v", i, len(prefix), string(line)))
		}
		lines[i] = line[len(prefix):]
	}
	return typed.YAMLObject(bytes.Join(lines, []byte{'\n'}))
}

func (s *State) checkInit(version fieldpath.APIVersion) error {
	if s.Live == nil {
		obj, err := s.Parser.Type(string(version)).FromUnstructured(nil)
		if err != nil {
			return fmt.Errorf("failed to create new empty object: %v", err)
		}
		s.Live = obj
	}
	return nil
}

func (s *State) UpdateObject(tv *typed.TypedValue, version fieldpath.APIVersion, manager string) error {
	err := s.checkInit(version)
	if err != nil {
		return err
	}
	s.Live, err = s.Updater.Converter.Convert(s.Live, version)
	if err != nil {
		return err
	}
	newObj, managers, err := s.Updater.Update(s.Live, tv, version, s.Managers, manager)
	if err != nil {
		return err
	}
	s.Live = newObj
	s.Managers = managers

	return nil
}

// Update the current state with the passed in object
func (s *State) Update(obj typed.YAMLObject, version fieldpath.APIVersion, manager string) error {
	tv, err := s.Parser.Type(string(version)).FromYAML(FixTabsOrDie(obj))
	if err != nil {
		return err
	}
	return s.UpdateObject(tv, version, manager)
}

func (s *State) ApplyObject(tv *typed.TypedValue, version fieldpath.APIVersion, manager string, force bool) error {
	err := s.checkInit(version)
	if err != nil {
		return err
	}
	s.Live, err = s.Updater.Converter.Convert(s.Live, version)
	if err != nil {
		return err
	}
	new, managers, err := s.Updater.Apply(s.Live, tv, version, s.Managers, manager, force)
	if err != nil {
		return err
	}
	s.Managers = managers
	if new != nil {
		s.Live = new
	}
	return nil
}

// Apply the passed in object to the current state
func (s *State) Apply(obj typed.YAMLObject, version fieldpath.APIVersion, manager string, force bool) error {
	tv, err := s.Parser.Type(string(version)).FromYAML(FixTabsOrDie(obj))
	if err != nil {
		return err
	}
	return s.ApplyObject(tv, version, manager, force)
}

// CompareLive takes a YAML string and returns the comparison with the
// current live object or an error.
func (s *State) CompareLive(obj typed.YAMLObject, version fieldpath.APIVersion) (*typed.Comparison, error) {
	obj = FixTabsOrDie(obj)
	if err := s.checkInit(version); err != nil {
		return nil, err
	}
	tv, err := s.Parser.Type(string(version)).FromYAML(obj)
	if err != nil {
		return nil, err
	}
	live, err := s.Updater.Converter.Convert(s.Live, version)
	if err != nil {
		return nil, err
	}
	return live.Compare(tv)
}

// dummyConverter doesn't convert, it just returns the same exact object, as long as a version is provided.
type dummyConverter struct{}

var _ merge.Converter = dummyConverter{}

// Convert returns the object given in input, not doing any conversion.
func (dummyConverter) Convert(v *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	if len(version) == 0 {
		return nil, fmt.Errorf("cannot convert to invalid version: %q", version)
	}
	return v, nil
}

func (dummyConverter) IsMissingVersionError(err error) bool {
	return false
}

// Operation is a step that will run when building a table-driven test.
type Operation interface {
	run(*State) error
	preprocess(Parser) (Operation, error)
}

func hasConflict(conflicts merge.Conflicts, conflict merge.Conflict) bool {
	for i := range conflicts {
		if conflict.Equals(conflicts[i]) {
			return true
		}
	}
	return false
}

func addedConflicts(one, other merge.Conflicts) merge.Conflicts {
	added := merge.Conflicts{}
	for _, conflict := range other {
		if !hasConflict(one, conflict) {
			added = append(added, conflict)
		}
	}
	return added
}

// Apply is a type of operation. It is a non-forced apply run by a
// manager with a given object. Since non-forced apply operation can
// conflict, the user can specify the expected conflicts. If conflicts
// don't match, an error will occur.
type Apply struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     typed.YAMLObject
	Conflicts  merge.Conflicts
}

var _ Operation = &Apply{}

func (a Apply) run(state *State) error {
	p, err := a.preprocess(state.Parser)
	if err != nil {
		return err
	}
	return p.run(state)
}

func (a Apply) preprocess(parser Parser) (Operation, error) {
	tv, err := parser.Type(string(a.APIVersion)).FromYAML(FixTabsOrDie(a.Object))
	if err != nil {
		return nil, err
	}
	return ApplyObject{
		Manager:    a.Manager,
		APIVersion: a.APIVersion,
		Object:     tv,
		Conflicts:  a.Conflicts,
	}, nil
}

type ApplyObject struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     *typed.TypedValue
	Conflicts  merge.Conflicts
}

var _ Operation = &ApplyObject{}

func (a ApplyObject) run(state *State) error {
	err := state.ApplyObject(a.Object, a.APIVersion, a.Manager, false)
	if err != nil {
		if _, ok := err.(merge.Conflicts); !ok || a.Conflicts == nil {
			return err
		}
	}
	if a.Conflicts != nil {
		conflicts := merge.Conflicts{}
		if err != nil {
			conflicts = err.(merge.Conflicts)
		}
		if len(addedConflicts(a.Conflicts, conflicts)) != 0 || len(addedConflicts(conflicts, a.Conflicts)) != 0 {
			return fmt.Errorf("Expected conflicts:\n%v\ngot\n%v\nadded:\n%v\nremoved:\n%v",
				a.Conflicts.Error(),
				conflicts.Error(),
				addedConflicts(a.Conflicts, conflicts).Error(),
				addedConflicts(conflicts, a.Conflicts).Error(),
			)
		}
	}
	return nil
}

func (a ApplyObject) preprocess(parser Parser) (Operation, error) {
	return a, nil
}

// ForceApply is a type of operation. It is a forced-apply run by a
// manager with a given object. Any error will be returned.
type ForceApply struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     typed.YAMLObject
}

var _ Operation = &ForceApply{}

func (f ForceApply) run(state *State) error {
	return state.Apply(f.Object, f.APIVersion, f.Manager, true)
}

func (f ForceApply) preprocess(parser Parser) (Operation, error) {
	tv, err := parser.Type(string(f.APIVersion)).FromYAML(FixTabsOrDie(f.Object))
	if err != nil {
		return nil, err
	}
	return ForceApplyObject{
		Manager:    f.Manager,
		APIVersion: f.APIVersion,
		Object:     tv,
	}, nil
}

// ForceApplyObject is a type of operation. It is a forced-apply run by
// a manager with a given object. Any error will be returned.
type ForceApplyObject struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     *typed.TypedValue
}

var _ Operation = &ForceApplyObject{}

func (f ForceApplyObject) run(state *State) error {
	return state.ApplyObject(f.Object, f.APIVersion, f.Manager, true)
}

func (f ForceApplyObject) preprocess(parser Parser) (Operation, error) {
	return f, nil
}

// ExtractApply is a type of operation. It simulates extracting an object
// the state based on the manager you have applied with, merging the
// apply object with that extracted object and reapplying that.
type ExtractApply struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     typed.YAMLObject
}

var _ Operation = &ExtractApply{}

func (e ExtractApply) run(state *State) error {
	p, err := e.preprocess(state.Parser)
	if err != nil {
		return err
	}
	return p.run(state)
}

func (e ExtractApply) preprocess(parser Parser) (Operation, error) {

	tv, err := parser.Type(string(e.APIVersion)).FromYAML(FixTabsOrDie(e.Object))
	if err != nil {
		return nil, err
	}
	return ExtractApplyObject{
		Manager:    e.Manager,
		APIVersion: e.APIVersion,
		Object:     tv,
	}, nil
}

type ExtractApplyObject struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     *typed.TypedValue
}

var _ Operation = &ExtractApplyObject{}

func (e ExtractApplyObject) run(state *State) error {
	if state.Live == nil {
		return state.ApplyObject(e.Object, e.APIVersion, e.Manager, true)
	}
	// Get object from state and convert it to current APIVersion
	current, err := state.Updater.Converter.Convert(state.Live, e.APIVersion)
	if err != nil {
		return err
	}
	// Get set based on the manager you've applied with
	set := fieldpath.NewSet()
	mgr := state.Managers[e.Manager]
	if mgr != nil {
		// we cannot extract a set that is for a different version
		if mgr.APIVersion() != e.APIVersion {
			return fmt.Errorf("existing managed fieldpath set APIVersion (%s) differs from desired (%s), unable to extract", mgr.APIVersion(), e.APIVersion)
		}
		// trying to extract the fieldSet directly will return everything
		// under the first path in the set, so we must filter out all
		// the non-leaf nodes from the fieldSet
		set = mgr.Set().Leaves()
	}
	// ExtractFields from the state object based on the set
	extracted := current.ExtractItems(set)
	// Merge ApplyObject on top of the extracted object
	obj, err := extracted.Merge(e.Object)
	if err != nil {
		return err
	}
	// Reapply that to the state
	return state.ApplyObject(obj, e.APIVersion, e.Manager, true)
}

func (e ExtractApplyObject) preprocess(parser Parser) (Operation, error) {
	return e, nil
}

// Update is a type of operation. It is a controller type of
// update. Errors are passed along.
type Update struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     typed.YAMLObject
}

var _ Operation = &Update{}

func (u Update) run(state *State) error {
	return state.Update(u.Object, u.APIVersion, u.Manager)
}

func (u Update) preprocess(parser Parser) (Operation, error) {
	tv, err := parser.Type(string(u.APIVersion)).FromYAML(FixTabsOrDie(u.Object))
	if err != nil {
		return nil, err
	}
	return UpdateObject{
		Manager:    u.Manager,
		APIVersion: u.APIVersion,
		Object:     tv,
	}, nil
}

// UpdateObject is a type of operation. It is a controller type of
// update. Errors are passed along.
type UpdateObject struct {
	Manager    string
	APIVersion fieldpath.APIVersion
	Object     *typed.TypedValue
}

var _ Operation = &Update{}

func (u UpdateObject) run(state *State) error {
	return state.UpdateObject(u.Object, u.APIVersion, u.Manager)
}

func (f UpdateObject) preprocess(parser Parser) (Operation, error) {
	return f, nil
}

// ChangeParser is a type of operation. It simulates making changes a schema without versioning
// the schema. This can be used to test the behavior of making backward compatible schema changes,
// e.g. setting "elementRelationship: atomic" on an existing struct. It also may be used to ensure
// that backward incompatible changes are detected appropriately.
type ChangeParser struct {
	Parser *typed.Parser
}

var _ Operation = &ChangeParser{}

func (cs ChangeParser) run(state *State) error {
	state.Parser = cs.Parser
	// Swap the schema in for use with the live object so it merges.
	// If the schema is incompatible, this will fail validation.

	liveWithNewSchema, err := typed.AsTyped(state.Live.AsValue(), &cs.Parser.Schema, state.Live.TypeRef())
	if err != nil {
		return err
	}
	state.Live = liveWithNewSchema
	return nil
}

func (cs ChangeParser) preprocess(_ Parser) (Operation, error) {
	return cs, nil
}

// TestCase is the list of operations that need to be run, as well as
// the object/managedfields as they are supposed to look like after all
// the operations have been successfully performed. If Object/Managed is
// not specified, then the comparison is not performed (any object or
// managed field will pass). Any error (conflicts aside) happen while
// running the operation, that error will be returned right away.
type TestCase struct {
	// Ops is the list of operations to run sequentially
	Ops []Operation
	// Object, if not empty, is the object as it's expected to
	// be after all the operations are run.
	Object typed.YAMLObject
	// APIVersion should be set if the object is non-empty and
	// describes the version of the object to compare to.
	APIVersion fieldpath.APIVersion
	// Managed, if not nil, is the ManagedFields as expected
	// after all operations are run.
	Managed fieldpath.ManagedFields
	// Set to true if the test case needs the union behavior enabled.
	RequiresUnions bool
	// IgnoredFields containing the set to ignore for every version
	IgnoredFields map[fieldpath.APIVersion]*fieldpath.Set
}

// Test runs the test-case using the given parser and a dummy converter.
func (tc TestCase) Test(parser Parser) error {
	return tc.TestWithConverter(parser, &dummyConverter{})
}

// Bench runs the test-case using the given parser and a dummy converter, but
// doesn't check exit conditions--see the comment for BenchWithConverter.
func (tc TestCase) Bench(parser Parser) error {
	return tc.BenchWithConverter(parser, &dummyConverter{})
}

// Preprocess all the operations by parsing the yaml before-hand.
func (tc TestCase) PreprocessOperations(parser Parser) error {
	for i := range tc.Ops {
		op, err := tc.Ops[i].preprocess(parser)
		if err != nil {
			return err
		}
		tc.Ops[i] = op
	}
	return nil
}

// BenchWithConverter runs the test-case using the given parser and converter,
// but doesn't do any comparison operations aftewards; you should probably run
// TestWithConverter once and reset the benchmark, to make sure the test case
// actually passes..
func (tc TestCase) BenchWithConverter(parser Parser, converter merge.Converter) error {
	state := State{
		Updater: &merge.Updater{Converter: converter, IgnoredFields: tc.IgnoredFields},
		Parser:  parser,
	}
	if tc.RequiresUnions {
		state.Updater.EnableUnionFeature()
	}
	// We currently don't have any test that converts, we can take
	// care of that later.
	for i, ops := range tc.Ops {
		err := ops.run(&state)
		if err != nil {
			return fmt.Errorf("failed operation %d: %v", i, err)
		}
	}
	return nil
}

// TestWithConverter runs the test-case using the given parser and converter.
func (tc TestCase) TestWithConverter(parser Parser, converter merge.Converter) error {
	state := State{
		Updater: &merge.Updater{Converter: converter, IgnoredFields: tc.IgnoredFields},
		Parser:  parser,
	}
	if tc.RequiresUnions {
		state.Updater.EnableUnionFeature()
	}
	for i, ops := range tc.Ops {
		err := ops.run(&state)
		if err != nil {
			return fmt.Errorf("failed operation %d: %v", i, err)
		}
	}

	// If LastObject was specified, compare it with LiveState
	if tc.Object != typed.YAMLObject("") {
		comparison, err := state.CompareLive(tc.Object, tc.APIVersion)
		if err != nil {
			return fmt.Errorf("failed to compare live with config: %v", err)
		}
		if !comparison.IsSame() {
			return fmt.Errorf("expected live and config to be the same:\n%v\nConfig: %v\n", comparison, value.ToString(state.Live.AsValue()))
		}
	}

	if tc.Managed != nil {
		if diff := state.Managers.Difference(tc.Managed); len(diff) != 0 {
			return fmt.Errorf("expected Managers to be:\n%v\ngot:\n%v", tc.Managed, state.Managers)
		}
	}

	// Fail if any empty sets are present in the managers
	for manager, set := range state.Managers {
		if set.Set().Empty() {
			return fmt.Errorf("expected Managers to have no empty sets, but found one managed by %v", manager)
		}
	}

	if !tc.RequiresUnions {
		// Re-run the test with unions on.
		tc2 := tc
		tc2.RequiresUnions = true
		err := tc2.TestWithConverter(parser, converter)
		if err != nil {
			return fmt.Errorf("fails if unions are on: %v", err)
		}
	}

	return nil
}
