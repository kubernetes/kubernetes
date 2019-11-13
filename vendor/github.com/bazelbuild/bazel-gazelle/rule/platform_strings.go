/* Copyright 2017 The Bazel Authors. All rights reserved.

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

package rule

import (
	"sort"
	"strings"

	bzl "github.com/bazelbuild/buildtools/build"
)

// PlatformStrings contains a set of strings associated with a buildable
// target in a package. This is used to store source file names,
// import paths, and flags.
//
// Strings are stored in four sets: generic strings, OS-specific strings,
// arch-specific strings, and OS-and-arch-specific strings. A string may not
// be duplicated within a list or across sets; however, a string may appear
// in more than one list within a set (e.g., in "linux" and "windows" within
// the OS set). Strings within each list should be sorted, though this may
// not be relied upon.
//
// DEPRECATED: do not use outside language/go. This type is Go-specific and
// should be moved to the Go extension.
type PlatformStrings struct {
	// Generic is a list of strings not specific to any platform.
	Generic []string

	// OS is a map from OS name (anything in KnownOSs) to
	// OS-specific strings.
	OS map[string][]string

	// Arch is a map from architecture name (anything in KnownArchs) to
	// architecture-specific strings.
	Arch map[string][]string

	// Platform is a map from platforms to OS and architecture-specific strings.
	Platform map[Platform][]string
}

// HasExt returns whether this set contains a file with the given extension.
func (ps *PlatformStrings) HasExt(ext string) bool {
	return ps.firstExtFile(ext) != ""
}

func (ps *PlatformStrings) IsEmpty() bool {
	return len(ps.Generic) == 0 && len(ps.OS) == 0 && len(ps.Arch) == 0 && len(ps.Platform) == 0
}

// Flat returns all the strings in the set, sorted and de-duplicated.
func (ps *PlatformStrings) Flat() []string {
	unique := make(map[string]struct{})
	for _, s := range ps.Generic {
		unique[s] = struct{}{}
	}
	for _, ss := range ps.OS {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	for _, ss := range ps.Arch {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	for _, ss := range ps.Platform {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	flat := make([]string, 0, len(unique))
	for s := range unique {
		flat = append(flat, s)
	}
	sort.Strings(flat)
	return flat
}

func (ps *PlatformStrings) firstExtFile(ext string) string {
	for _, f := range ps.Generic {
		if strings.HasSuffix(f, ext) {
			return f
		}
	}
	for _, fs := range ps.OS {
		for _, f := range fs {
			if strings.HasSuffix(f, ext) {
				return f
			}
		}
	}
	for _, fs := range ps.Arch {
		for _, f := range fs {
			if strings.HasSuffix(f, ext) {
				return f
			}
		}
	}
	for _, fs := range ps.Platform {
		for _, f := range fs {
			if strings.HasSuffix(f, ext) {
				return f
			}
		}
	}
	return ""
}

// Map applies a function that processes individual strings to the strings
// in "ps" and returns a new PlatformStrings with the result. Empty strings
// returned by the function are dropped.
func (ps *PlatformStrings) Map(f func(s string) (string, error)) (PlatformStrings, []error) {
	var errors []error
	mapSlice := func(ss []string) ([]string, error) {
		rs := make([]string, 0, len(ss))
		for _, s := range ss {
			if r, err := f(s); err != nil {
				errors = append(errors, err)
			} else if r != "" {
				rs = append(rs, r)
			}
		}
		return rs, nil
	}
	result, _ := ps.MapSlice(mapSlice)
	return result, errors
}

// MapSlice applies a function that processes slices of strings to the strings
// in "ps" and returns a new PlatformStrings with the results.
func (ps *PlatformStrings) MapSlice(f func([]string) ([]string, error)) (PlatformStrings, []error) {
	var errors []error

	mapSlice := func(ss []string) []string {
		rs, err := f(ss)
		if err != nil {
			errors = append(errors, err)
			return nil
		}
		return rs
	}

	mapStringMap := func(m map[string][]string) map[string][]string {
		if m == nil {
			return nil
		}
		rm := make(map[string][]string)
		for k, ss := range m {
			ss = mapSlice(ss)
			if len(ss) > 0 {
				rm[k] = ss
			}
		}
		if len(rm) == 0 {
			return nil
		}
		return rm
	}

	mapPlatformMap := func(m map[Platform][]string) map[Platform][]string {
		if m == nil {
			return nil
		}
		rm := make(map[Platform][]string)
		for k, ss := range m {
			ss = mapSlice(ss)
			if len(ss) > 0 {
				rm[k] = ss
			}
		}
		if len(rm) == 0 {
			return nil
		}
		return rm
	}

	result := PlatformStrings{
		Generic:  mapSlice(ps.Generic),
		OS:       mapStringMap(ps.OS),
		Arch:     mapStringMap(ps.Arch),
		Platform: mapPlatformMap(ps.Platform),
	}
	return result, errors
}

func (ps PlatformStrings) BzlExpr() bzl.Expr {
	var pieces []bzl.Expr
	if len(ps.Generic) > 0 {
		pieces = append(pieces, ExprFromValue(ps.Generic))
	}
	if len(ps.OS) > 0 {
		pieces = append(pieces, platformStringsOSArchDictExpr(ps.OS))
	}
	if len(ps.Arch) > 0 {
		pieces = append(pieces, platformStringsOSArchDictExpr(ps.Arch))
	}
	if len(ps.Platform) > 0 {
		pieces = append(pieces, platformStringsPlatformDictExpr(ps.Platform))
	}
	if len(pieces) == 0 {
		return &bzl.ListExpr{}
	} else if len(pieces) == 1 {
		return pieces[0]
	} else {
		e := pieces[0]
		if list, ok := e.(*bzl.ListExpr); ok {
			list.ForceMultiLine = true
		}
		for _, piece := range pieces[1:] {
			e = &bzl.BinaryExpr{X: e, Y: piece, Op: "+"}
		}
		return e
	}
}

func platformStringsOSArchDictExpr(m map[string][]string) bzl.Expr {
	s := make(SelectStringListValue)
	for key, value := range m {
		s["@io_bazel_rules_go//go/platform:"+key] = value
	}
	s["//conditions:default"] = nil
	return s.BzlExpr()
}

func platformStringsPlatformDictExpr(m map[Platform][]string) bzl.Expr {
	s := make(SelectStringListValue)
	for key, value := range m {
		s["@io_bazel_rules_go//go/platform:"+key.String()] = value
	}
	s["//conditions:default"] = nil
	return s.BzlExpr()
}
