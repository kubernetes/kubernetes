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

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
	"sigs.k8s.io/structured-merge-diff/typed"
)

// State of the current test in terms of live object. One can check at
// any time that Live and Managers match the expectations.
type State struct {
	Live     *typed.TypedValue
	Parser   typed.ParseableType
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
			panic(fmt.Errorf("line %d doesn't start with expected number (%d) of tabs: %v", i, len(prefix), line))
		}
		lines[i] = line[len(prefix):]
	}
	return typed.YAMLObject(bytes.Join(lines, []byte{'\n'}))
}

func (s *State) checkInit() error {
	if s.Live == nil {
		obj, err := s.Parser.FromUnstructured(nil)
		if err != nil {
			return fmt.Errorf("failed to create new empty object: %v", err)
		}
		s.Live = obj
	}
	return nil
}

func (s *State) UpdateObject(tv *typed.TypedValue, version fieldpath.APIVersion, manager string) error {
	err := s.checkInit()
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
	tv, err := s.Parser.FromYAML(FixTabsOrDie(obj))
	if err != nil {
		return err
	}
	return s.UpdateObject(tv, version, manager)
}

func (s *State) ApplyObject(tv *typed.TypedValue, version fieldpath.APIVersion, manager string, force bool) error {
	err := s.checkInit()
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
	s.Live = new
	s.Managers = managers

	return nil
}

// Apply the passed in object to the current state
func (s *State) Apply(obj typed.YAMLObject, version fieldpath.APIVersion, manager string, force bool) error {
	tv, err := s.Parser.FromYAML(FixTabsOrDie(obj))
	if err != nil {
		return err
	}
	return s.ApplyObject(tv, version, manager, force)
}

// CompareLive takes a YAML string and returns the comparison with the
// current live object or an error.
func (s *State) CompareLive(obj typed.YAMLObject) (*typed.Comparison, error) {
	obj = FixTabsOrDie(obj)
	if err := s.checkInit(); err != nil {
		return nil, err
	}
	tv, err := s.Parser.FromYAML(obj)
	if err != nil {
		return nil, err
	}
	return s.Live.Compare(tv)
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
	preprocess(typed.ParseableType) (Operation, error)
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

func (a Apply) preprocess(parser typed.ParseableType) (Operation, error) {
	tv, err := parser.FromYAML(FixTabsOrDie(a.Object))
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

func (a ApplyObject) preprocess(parser typed.ParseableType) (Operation, error) {
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

func (f ForceApply) preprocess(parser typed.ParseableType) (Operation, error) {
	tv, err := parser.FromYAML(FixTabsOrDie(f.Object))
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

func (f ForceApplyObject) preprocess(parser typed.ParseableType) (Operation, error) {
	return f, nil
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

func (u Update) preprocess(parser typed.ParseableType) (Operation, error) {
	tv, err := parser.FromYAML(FixTabsOrDie(u.Object))
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

func (f UpdateObject) preprocess(parser typed.ParseableType) (Operation, error) {
	return f, nil
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
	// Managed, if not nil, is the ManagedFields as expected
	// after all operations are run.
	Managed fieldpath.ManagedFields
	// Set to true if the test case needs the union behavior enabled.
	RequiresUnions bool
}

// Test runs the test-case using the given parser and a dummy converter.
func (tc TestCase) Test(parser typed.ParseableType) error {
	return tc.TestWithConverter(parser, &dummyConverter{})
}

// Bench runs the test-case using the given parser and a dummy converter, but
// doesn't check exit conditions--see the comment for BenchWithConverter.
func (tc TestCase) Bench(parser typed.ParseableType) error {
	return tc.BenchWithConverter(parser, &dummyConverter{})
}

// Preprocess all the operations by parsing the yaml before-hand.
func (tc TestCase) PreprocessOperations(parser typed.ParseableType) error {
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
func (tc TestCase) BenchWithConverter(parser typed.ParseableType, converter merge.Converter) error {
	state := State{
		Updater: &merge.Updater{Converter: converter},
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
func (tc TestCase) TestWithConverter(parser typed.ParseableType, converter merge.Converter) error {
	state := State{
		Updater: &merge.Updater{Converter: converter},
		Parser:  parser,
	}
	if tc.RequiresUnions {
		state.Updater.EnableUnionFeature()
	} else {
		// Also test it with unions on.
		tc2 := tc
		tc2.RequiresUnions = true
		err := tc2.TestWithConverter(parser, converter)
		if err != nil {
			return fmt.Errorf("fails if unions are on: %v", err)
		}
	}
	// We currently don't have any test that converts, we can take
	// care of that later.
	for i, ops := range tc.Ops {
		err := ops.run(&state)
		if err != nil {
			return fmt.Errorf("failed operation %d: %v", i, err)
		}
	}

	// If LastObject was specified, compare it with LiveState
	if tc.Object != typed.YAMLObject("") {
		comparison, err := state.CompareLive(tc.Object)
		if err != nil {
			return fmt.Errorf("failed to compare live with config: %v", err)
		}
		if !comparison.IsSame() {
			return fmt.Errorf("expected live and config to be the same:\n%v", comparison)
		}
	}

	if tc.Managed != nil {
		if diff := state.Managers.Difference(tc.Managed); len(diff) != 0 {
			return fmt.Errorf("expected Managers to be %v, got %v", tc.Managed, state.Managers)
		}
	}

	// Fail if any empty sets are present in the managers
	for manager, set := range state.Managers {
		if set.Set().Empty() {
			return fmt.Errorf("expected Managers to have no empty sets, but found one managed by %v", manager)
		}
	}

	return nil
}
