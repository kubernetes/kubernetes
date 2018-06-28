// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

// StringDiff represents a modified string value.
// * Added: Previous = nil, Current != nil
// * Deleted: Previous != nil, Current = nil
// * Modified: Previous != nil, Current != nil
// * No Change: Previous = Current, or a nil pointer
type StringDiff struct {
	Previous string
	Current  string
}

func (diff *StringDiff) String() string {
	if diff == nil {
		return ""
	}

	if diff.Previous == "" && diff.Current != "" {
		return fmt.Sprintf("+ %s", diff.Current)
	}

	if diff.Previous != "" && diff.Current == "" {
		return fmt.Sprintf("- %s", diff.Previous)
	}

	if diff.Previous != diff.Current {
		return fmt.Sprintf("%s -> %s", diff.Previous, diff.Current)
	}

	return diff.Current
}

// LockDiff is the set of differences between an existing lock file and an updated lock file.
// Fields are only populated when there is a difference, otherwise they are empty.
type LockDiff struct {
	HashDiff *StringDiff
	Add      []LockedProjectDiff
	Remove   []LockedProjectDiff
	Modify   []LockedProjectDiff
}

// LockedProjectDiff contains the before and after snapshot of a project reference.
// Fields are only populated when there is a difference, otherwise they are empty.
type LockedProjectDiff struct {
	Name     ProjectRoot
	Source   *StringDiff
	Version  *StringDiff
	Branch   *StringDiff
	Revision *StringDiff
	Packages []StringDiff
}

// DiffLocks compares two locks and identifies the differences between them.
// Returns nil if there are no differences.
func DiffLocks(l1 Lock, l2 Lock) *LockDiff {
	// Default nil locks to empty locks, so that we can still generate a diff
	if l1 == nil {
		l1 = &SimpleLock{}
	}
	if l2 == nil {
		l2 = &SimpleLock{}
	}

	p1, p2 := l1.Projects(), l2.Projects()

	p1 = sortedLockedProjects(p1)
	p2 = sortedLockedProjects(p2)

	diff := LockDiff{}

	h1 := hex.EncodeToString(l1.InputsDigest())
	h2 := hex.EncodeToString(l2.InputsDigest())
	if h1 != h2 {
		diff.HashDiff = &StringDiff{Previous: h1, Current: h2}
	}

	var i2next int
	for i1 := 0; i1 < len(p1); i1++ {
		lp1 := p1[i1]
		pr1 := lp1.pi.ProjectRoot

		var matched bool
		for i2 := i2next; i2 < len(p2); i2++ {
			lp2 := p2[i2]
			pr2 := lp2.pi.ProjectRoot

			switch strings.Compare(string(pr1), string(pr2)) {
			case 0: // Found a matching project
				matched = true
				pdiff := DiffProjects(lp1, lp2)
				if pdiff != nil {
					diff.Modify = append(diff.Modify, *pdiff)
				}
				i2next = i2 + 1 // Don't evaluate to this again
			case +1: // Found a new project
				add := buildLockedProjectDiff(lp2)
				diff.Add = append(diff.Add, add)
				i2next = i2 + 1 // Don't evaluate to this again
				continue        // Keep looking for a matching project
			case -1: // Project has been removed, handled below
				continue
			}

			break // Done evaluating this project, move onto the next
		}

		if !matched {
			remove := buildLockedProjectDiff(lp1)
			diff.Remove = append(diff.Remove, remove)
		}
	}

	// Anything that still hasn't been evaluated are adds
	for i2 := i2next; i2 < len(p2); i2++ {
		lp2 := p2[i2]
		add := buildLockedProjectDiff(lp2)
		diff.Add = append(diff.Add, add)
	}

	if diff.HashDiff == nil && len(diff.Add) == 0 && len(diff.Remove) == 0 && len(diff.Modify) == 0 {
		return nil // The locks are the equivalent
	}
	return &diff
}

func buildLockedProjectDiff(lp LockedProject) LockedProjectDiff {
	s2 := lp.pi.Source
	r2, b2, v2 := VersionComponentStrings(lp.Version())

	var rev, version, branch, source *StringDiff
	if s2 != "" {
		source = &StringDiff{Previous: s2, Current: s2}
	}
	if r2 != "" {
		rev = &StringDiff{Previous: r2, Current: r2}
	}
	if b2 != "" {
		branch = &StringDiff{Previous: b2, Current: b2}
	}
	if v2 != "" {
		version = &StringDiff{Previous: v2, Current: v2}
	}

	add := LockedProjectDiff{
		Name:     lp.pi.ProjectRoot,
		Source:   source,
		Revision: rev,
		Version:  version,
		Branch:   branch,
		Packages: make([]StringDiff, len(lp.Packages())),
	}
	for i, pkg := range lp.Packages() {
		add.Packages[i] = StringDiff{Previous: pkg, Current: pkg}
	}
	return add
}

// DiffProjects compares two projects and identifies the differences between them.
// Returns nil if there are no differences
func DiffProjects(lp1 LockedProject, lp2 LockedProject) *LockedProjectDiff {
	diff := LockedProjectDiff{Name: lp1.pi.ProjectRoot}

	s1 := lp1.pi.Source
	s2 := lp2.pi.Source
	if s1 != s2 {
		diff.Source = &StringDiff{Previous: s1, Current: s2}
	}

	r1, b1, v1 := VersionComponentStrings(lp1.Version())
	r2, b2, v2 := VersionComponentStrings(lp2.Version())
	if r1 != r2 {
		diff.Revision = &StringDiff{Previous: r1, Current: r2}
	}
	if b1 != b2 {
		diff.Branch = &StringDiff{Previous: b1, Current: b2}
	}
	if v1 != v2 {
		diff.Version = &StringDiff{Previous: v1, Current: v2}
	}

	p1 := lp1.Packages()
	p2 := lp2.Packages()
	if !sort.StringsAreSorted(p1) {
		p1 = make([]string, len(p1))
		copy(p1, lp1.Packages())
		sort.Strings(p1)
	}
	if !sort.StringsAreSorted(p2) {
		p2 = make([]string, len(p2))
		copy(p2, lp2.Packages())
		sort.Strings(p2)
	}

	var i2next int
	for i1 := 0; i1 < len(p1); i1++ {
		pkg1 := p1[i1]

		var matched bool
		for i2 := i2next; i2 < len(p2); i2++ {
			pkg2 := p2[i2]

			switch strings.Compare(pkg1, pkg2) {
			case 0: // Found matching package
				matched = true
				i2next = i2 + 1 // Don't evaluate to this again
			case +1: // Found a new package
				add := StringDiff{Current: pkg2}
				diff.Packages = append(diff.Packages, add)
				i2next = i2 + 1 // Don't evaluate to this again
				continue        // Keep looking for a match
			case -1: // Package has been removed (handled below)
				continue
			}

			break // Done evaluating this package, move onto the next
		}

		if !matched {
			diff.Packages = append(diff.Packages, StringDiff{Previous: pkg1})
		}
	}

	// Anything that still hasn't been evaluated are adds
	for i2 := i2next; i2 < len(p2); i2++ {
		pkg2 := p2[i2]
		add := StringDiff{Current: pkg2}
		diff.Packages = append(diff.Packages, add)
	}

	if diff.Source == nil && diff.Version == nil && diff.Revision == nil && len(diff.Packages) == 0 {
		return nil // The projects are equivalent
	}
	return &diff
}
