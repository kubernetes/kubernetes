// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/google/go-cmp/cmp/internal/function"
)

// Option configures for specific behavior of Equal and Diff. In particular,
// the fundamental Option functions (Ignore, Transformer, and Comparer),
// configure how equality is determined.
//
// The fundamental options may be composed with filters (FilterPath and
// FilterValues) to control the scope over which they are applied.
//
// The cmp/cmpopts package provides helper functions for creating options that
// may be used with Equal and Diff.
type Option interface {
	// filter applies all filters and returns the option that remains.
	// Each option may only read s.curPath and call s.callTTBFunc.
	//
	// An Options is returned only if multiple comparers or transformers
	// can apply simultaneously and will only contain values of those types
	// or sub-Options containing values of those types.
	filter(s *state, t reflect.Type, vx, vy reflect.Value) applicableOption
}

// applicableOption represents the following types:
//
//	Fundamental: ignore | validator | *comparer | *transformer
//	Grouping:    Options
type applicableOption interface {
	Option

	// apply executes the option, which may mutate s or panic.
	apply(s *state, vx, vy reflect.Value)
}

// coreOption represents the following types:
//
//	Fundamental: ignore | validator | *comparer | *transformer
//	Filters:     *pathFilter | *valuesFilter
type coreOption interface {
	Option
	isCore()
}

type core struct{}

func (core) isCore() {}

// Options is a list of Option values that also satisfies the Option interface.
// Helper comparison packages may return an Options value when packing multiple
// Option values into a single Option. When this package processes an Options,
// it will be implicitly expanded into a flat list.
//
// Applying a filter on an Options is equivalent to applying that same filter
// on all individual options held within.
type Options []Option

func (opts Options) filter(s *state, t reflect.Type, vx, vy reflect.Value) (out applicableOption) {
	for _, opt := range opts {
		switch opt := opt.filter(s, t, vx, vy); opt.(type) {
		case ignore:
			return ignore{} // Only ignore can short-circuit evaluation
		case validator:
			out = validator{} // Takes precedence over comparer or transformer
		case *comparer, *transformer, Options:
			switch out.(type) {
			case nil:
				out = opt
			case validator:
				// Keep validator
			case *comparer, *transformer, Options:
				out = Options{out, opt} // Conflicting comparers or transformers
			}
		}
	}
	return out
}

func (opts Options) apply(s *state, _, _ reflect.Value) {
	const warning = "ambiguous set of applicable options"
	const help = "consider using filters to ensure at most one Comparer or Transformer may apply"
	var ss []string
	for _, opt := range flattenOptions(nil, opts) {
		ss = append(ss, fmt.Sprint(opt))
	}
	set := strings.Join(ss, "\n\t")
	panic(fmt.Sprintf("%s at %#v:\n\t%s\n%s", warning, s.curPath, set, help))
}

func (opts Options) String() string {
	var ss []string
	for _, opt := range opts {
		ss = append(ss, fmt.Sprint(opt))
	}
	return fmt.Sprintf("Options{%s}", strings.Join(ss, ", "))
}

// FilterPath returns a new Option where opt is only evaluated if filter f
// returns true for the current Path in the value tree.
//
// This filter is called even if a slice element or map entry is missing and
// provides an opportunity to ignore such cases. The filter function must be
// symmetric such that the filter result is identical regardless of whether the
// missing value is from x or y.
//
// The option passed in may be an Ignore, Transformer, Comparer, Options, or
// a previously filtered Option.
func FilterPath(f func(Path) bool, opt Option) Option {
	if f == nil {
		panic("invalid path filter function")
	}
	if opt := normalizeOption(opt); opt != nil {
		return &pathFilter{fnc: f, opt: opt}
	}
	return nil
}

type pathFilter struct {
	core
	fnc func(Path) bool
	opt Option
}

func (f pathFilter) filter(s *state, t reflect.Type, vx, vy reflect.Value) applicableOption {
	if f.fnc(s.curPath) {
		return f.opt.filter(s, t, vx, vy)
	}
	return nil
}

func (f pathFilter) String() string {
	return fmt.Sprintf("FilterPath(%s, %v)", function.NameOf(reflect.ValueOf(f.fnc)), f.opt)
}

// FilterValues returns a new Option where opt is only evaluated if filter f,
// which is a function of the form "func(T, T) bool", returns true for the
// current pair of values being compared. If either value is invalid or
// the type of the values is not assignable to T, then this filter implicitly
// returns false.
//
// The filter function must be
// symmetric (i.e., agnostic to the order of the inputs) and
// deterministic (i.e., produces the same result when given the same inputs).
// If T is an interface, it is possible that f is called with two values with
// different concrete types that both implement T.
//
// The option passed in may be an Ignore, Transformer, Comparer, Options, or
// a previously filtered Option.
func FilterValues(f interface{}, opt Option) Option {
	v := reflect.ValueOf(f)
	if !function.IsType(v.Type(), function.ValueFilter) || v.IsNil() {
		panic(fmt.Sprintf("invalid values filter function: %T", f))
	}
	if opt := normalizeOption(opt); opt != nil {
		vf := &valuesFilter{fnc: v, opt: opt}
		if ti := v.Type().In(0); ti.Kind() != reflect.Interface || ti.NumMethod() > 0 {
			vf.typ = ti
		}
		return vf
	}
	return nil
}

type valuesFilter struct {
	core
	typ reflect.Type  // T
	fnc reflect.Value // func(T, T) bool
	opt Option
}

func (f valuesFilter) filter(s *state, t reflect.Type, vx, vy reflect.Value) applicableOption {
	if !vx.IsValid() || !vx.CanInterface() || !vy.IsValid() || !vy.CanInterface() {
		return nil
	}
	if (f.typ == nil || t.AssignableTo(f.typ)) && s.callTTBFunc(f.fnc, vx, vy) {
		return f.opt.filter(s, t, vx, vy)
	}
	return nil
}

func (f valuesFilter) String() string {
	return fmt.Sprintf("FilterValues(%s, %v)", function.NameOf(f.fnc), f.opt)
}

// Ignore is an Option that causes all comparisons to be ignored.
// This value is intended to be combined with FilterPath or FilterValues.
// It is an error to pass an unfiltered Ignore option to Equal.
func Ignore() Option { return ignore{} }

type ignore struct{ core }

func (ignore) isFiltered() bool                                                     { return false }
func (ignore) filter(_ *state, _ reflect.Type, _, _ reflect.Value) applicableOption { return ignore{} }
func (ignore) apply(s *state, _, _ reflect.Value)                                   { s.report(true, reportByIgnore) }
func (ignore) String() string                                                       { return "Ignore()" }

// validator is a sentinel Option type to indicate that some options could not
// be evaluated due to unexported fields, missing slice elements, or
// missing map entries. Both values are validator only for unexported fields.
type validator struct{ core }

func (validator) filter(_ *state, _ reflect.Type, vx, vy reflect.Value) applicableOption {
	if !vx.IsValid() || !vy.IsValid() {
		return validator{}
	}
	if !vx.CanInterface() || !vy.CanInterface() {
		return validator{}
	}
	return nil
}
func (validator) apply(s *state, vx, vy reflect.Value) {
	// Implies missing slice element or map entry.
	if !vx.IsValid() || !vy.IsValid() {
		s.report(vx.IsValid() == vy.IsValid(), 0)
		return
	}

	// Unable to Interface implies unexported field without visibility access.
	if !vx.CanInterface() || !vy.CanInterface() {
		help := "consider using a custom Comparer; if you control the implementation of type, you can also consider using an Exporter, AllowUnexported, or cmpopts.IgnoreUnexported"
		var name string
		if t := s.curPath.Index(-2).Type(); t.Name() != "" {
			// Named type with unexported fields.
			name = fmt.Sprintf("%q.%v", t.PkgPath(), t.Name()) // e.g., "path/to/package".MyType
			if _, ok := reflect.New(t).Interface().(error); ok {
				help = "consider using cmpopts.EquateErrors to compare error values"
			}
		} else {
			// Unnamed type with unexported fields. Derive PkgPath from field.
			var pkgPath string
			for i := 0; i < t.NumField() && pkgPath == ""; i++ {
				pkgPath = t.Field(i).PkgPath
			}
			name = fmt.Sprintf("%q.(%v)", pkgPath, t.String()) // e.g., "path/to/package".(struct { a int })
		}
		panic(fmt.Sprintf("cannot handle unexported field at %#v:\n\t%v\n%s", s.curPath, name, help))
	}

	panic("not reachable")
}

// identRx represents a valid identifier according to the Go specification.
const identRx = `[_\p{L}][_\p{L}\p{N}]*`

var identsRx = regexp.MustCompile(`^` + identRx + `(\.` + identRx + `)*$`)

// Transformer returns an Option that applies a transformation function that
// converts values of a certain type into that of another.
//
// The transformer f must be a function "func(T) R" that converts values of
// type T to those of type R and is implicitly filtered to input values
// assignable to T. The transformer must not mutate T in any way.
//
// To help prevent some cases of infinite recursive cycles applying the
// same transform to the output of itself (e.g., in the case where the
// input and output types are the same), an implicit filter is added such that
// a transformer is applicable only if that exact transformer is not already
// in the tail of the Path since the last non-Transform step.
// For situations where the implicit filter is still insufficient,
// consider using cmpopts.AcyclicTransformer, which adds a filter
// to prevent the transformer from being recursively applied upon itself.
//
// The name is a user provided label that is used as the Transform.Name in the
// transformation PathStep (and eventually shown in the Diff output).
// The name must be a valid identifier or qualified identifier in Go syntax.
// If empty, an arbitrary name is used.
func Transformer(name string, f interface{}) Option {
	v := reflect.ValueOf(f)
	if !function.IsType(v.Type(), function.Transformer) || v.IsNil() {
		panic(fmt.Sprintf("invalid transformer function: %T", f))
	}
	if name == "" {
		name = function.NameOf(v)
		if !identsRx.MatchString(name) {
			name = "λ" // Lambda-symbol as placeholder name
		}
	} else if !identsRx.MatchString(name) {
		panic(fmt.Sprintf("invalid name: %q", name))
	}
	tr := &transformer{name: name, fnc: reflect.ValueOf(f)}
	if ti := v.Type().In(0); ti.Kind() != reflect.Interface || ti.NumMethod() > 0 {
		tr.typ = ti
	}
	return tr
}

type transformer struct {
	core
	name string
	typ  reflect.Type  // T
	fnc  reflect.Value // func(T) R
}

func (tr *transformer) isFiltered() bool { return tr.typ != nil }

func (tr *transformer) filter(s *state, t reflect.Type, _, _ reflect.Value) applicableOption {
	for i := len(s.curPath) - 1; i >= 0; i-- {
		if t, ok := s.curPath[i].(Transform); !ok {
			break // Hit most recent non-Transform step
		} else if tr == t.trans {
			return nil // Cannot directly use same Transform
		}
	}
	if tr.typ == nil || t.AssignableTo(tr.typ) {
		return tr
	}
	return nil
}

func (tr *transformer) apply(s *state, vx, vy reflect.Value) {
	step := Transform{&transform{pathStep{typ: tr.fnc.Type().Out(0)}, tr}}
	vvx := s.callTRFunc(tr.fnc, vx, step)
	vvy := s.callTRFunc(tr.fnc, vy, step)
	step.vx, step.vy = vvx, vvy
	s.compareAny(step)
}

func (tr transformer) String() string {
	return fmt.Sprintf("Transformer(%s, %s)", tr.name, function.NameOf(tr.fnc))
}

// Comparer returns an Option that determines whether two values are equal
// to each other.
//
// The comparer f must be a function "func(T, T) bool" and is implicitly
// filtered to input values assignable to T. If T is an interface, it is
// possible that f is called with two values of different concrete types that
// both implement T.
//
// The equality function must be:
//   - Symmetric: equal(x, y) == equal(y, x)
//   - Deterministic: equal(x, y) == equal(x, y)
//   - Pure: equal(x, y) does not modify x or y
func Comparer(f interface{}) Option {
	v := reflect.ValueOf(f)
	if !function.IsType(v.Type(), function.Equal) || v.IsNil() {
		panic(fmt.Sprintf("invalid comparer function: %T", f))
	}
	cm := &comparer{fnc: v}
	if ti := v.Type().In(0); ti.Kind() != reflect.Interface || ti.NumMethod() > 0 {
		cm.typ = ti
	}
	return cm
}

type comparer struct {
	core
	typ reflect.Type  // T
	fnc reflect.Value // func(T, T) bool
}

func (cm *comparer) isFiltered() bool { return cm.typ != nil }

func (cm *comparer) filter(_ *state, t reflect.Type, _, _ reflect.Value) applicableOption {
	if cm.typ == nil || t.AssignableTo(cm.typ) {
		return cm
	}
	return nil
}

func (cm *comparer) apply(s *state, vx, vy reflect.Value) {
	eq := s.callTTBFunc(cm.fnc, vx, vy)
	s.report(eq, reportByFunc)
}

func (cm comparer) String() string {
	return fmt.Sprintf("Comparer(%s)", function.NameOf(cm.fnc))
}

// Exporter returns an Option that specifies whether Equal is allowed to
// introspect into the unexported fields of certain struct types.
//
// Users of this option must understand that comparing on unexported fields
// from external packages is not safe since changes in the internal
// implementation of some external package may cause the result of Equal
// to unexpectedly change. However, it may be valid to use this option on types
// defined in an internal package where the semantic meaning of an unexported
// field is in the control of the user.
//
// In many cases, a custom Comparer should be used instead that defines
// equality as a function of the public API of a type rather than the underlying
// unexported implementation.
//
// For example, the reflect.Type documentation defines equality to be determined
// by the == operator on the interface (essentially performing a shallow pointer
// comparison) and most attempts to compare *regexp.Regexp types are interested
// in only checking that the regular expression strings are equal.
// Both of these are accomplished using Comparers:
//
//	Comparer(func(x, y reflect.Type) bool { return x == y })
//	Comparer(func(x, y *regexp.Regexp) bool { return x.String() == y.String() })
//
// In other cases, the cmpopts.IgnoreUnexported option can be used to ignore
// all unexported fields on specified struct types.
func Exporter(f func(reflect.Type) bool) Option {
	if !supportExporters {
		panic("Exporter is not supported on purego builds")
	}
	return exporter(f)
}

type exporter func(reflect.Type) bool

func (exporter) filter(_ *state, _ reflect.Type, _, _ reflect.Value) applicableOption {
	panic("not implemented")
}

// AllowUnexported returns an Options that allows Equal to forcibly introspect
// unexported fields of the specified struct types.
//
// See Exporter for the proper use of this option.
func AllowUnexported(types ...interface{}) Option {
	m := make(map[reflect.Type]bool)
	for _, typ := range types {
		t := reflect.TypeOf(typ)
		if t.Kind() != reflect.Struct {
			panic(fmt.Sprintf("invalid struct type: %T", typ))
		}
		m[t] = true
	}
	return exporter(func(t reflect.Type) bool { return m[t] })
}

// Result represents the comparison result for a single node and
// is provided by cmp when calling Report (see Reporter).
type Result struct {
	_     [0]func() // Make Result incomparable
	flags resultFlags
}

// Equal reports whether the node was determined to be equal or not.
// As a special case, ignored nodes are considered equal.
func (r Result) Equal() bool {
	return r.flags&(reportEqual|reportByIgnore) != 0
}

// ByIgnore reports whether the node is equal because it was ignored.
// This never reports true if Equal reports false.
func (r Result) ByIgnore() bool {
	return r.flags&reportByIgnore != 0
}

// ByMethod reports whether the Equal method determined equality.
func (r Result) ByMethod() bool {
	return r.flags&reportByMethod != 0
}

// ByFunc reports whether a Comparer function determined equality.
func (r Result) ByFunc() bool {
	return r.flags&reportByFunc != 0
}

// ByCycle reports whether a reference cycle was detected.
func (r Result) ByCycle() bool {
	return r.flags&reportByCycle != 0
}

type resultFlags uint

const (
	_ resultFlags = (1 << iota) / 2

	reportEqual
	reportUnequal
	reportByIgnore
	reportByMethod
	reportByFunc
	reportByCycle
)

// Reporter is an Option that can be passed to Equal. When Equal traverses
// the value trees, it calls PushStep as it descends into each node in the
// tree and PopStep as it ascend out of the node. The leaves of the tree are
// either compared (determined to be equal or not equal) or ignored and reported
// as such by calling the Report method.
func Reporter(r interface {
	// PushStep is called when a tree-traversal operation is performed.
	// The PathStep itself is only valid until the step is popped.
	// The PathStep.Values are valid for the duration of the entire traversal
	// and must not be mutated.
	//
	// Equal always calls PushStep at the start to provide an operation-less
	// PathStep used to report the root values.
	//
	// Within a slice, the exact set of inserted, removed, or modified elements
	// is unspecified and may change in future implementations.
	// The entries of a map are iterated through in an unspecified order.
	PushStep(PathStep)

	// Report is called exactly once on leaf nodes to report whether the
	// comparison identified the node as equal, unequal, or ignored.
	// A leaf node is one that is immediately preceded by and followed by
	// a pair of PushStep and PopStep calls.
	Report(Result)

	// PopStep ascends back up the value tree.
	// There is always a matching pop call for every push call.
	PopStep()
}) Option {
	return reporter{r}
}

type reporter struct{ reporterIface }
type reporterIface interface {
	PushStep(PathStep)
	Report(Result)
	PopStep()
}

func (reporter) filter(_ *state, _ reflect.Type, _, _ reflect.Value) applicableOption {
	panic("not implemented")
}

// normalizeOption normalizes the input options such that all Options groups
// are flattened and groups with a single element are reduced to that element.
// Only coreOptions and Options containing coreOptions are allowed.
func normalizeOption(src Option) Option {
	switch opts := flattenOptions(nil, Options{src}); len(opts) {
	case 0:
		return nil
	case 1:
		return opts[0]
	default:
		return opts
	}
}

// flattenOptions copies all options in src to dst as a flat list.
// Only coreOptions and Options containing coreOptions are allowed.
func flattenOptions(dst, src Options) Options {
	for _, opt := range src {
		switch opt := opt.(type) {
		case nil:
			continue
		case Options:
			dst = flattenOptions(dst, opt)
		case coreOption:
			dst = append(dst, opt)
		default:
			panic(fmt.Sprintf("invalid option type: %T", opt))
		}
	}
	return dst
}
