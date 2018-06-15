// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package feedback

import (
	"encoding/hex"
	"fmt"
	"log"

	"github.com/golang/dep/gps"
)

const (
	// ConsTypeConstraint represents a constraint
	ConsTypeConstraint = "constraint"

	// ConsTypeHint represents a constraint type hint
	ConsTypeHint = "hint"

	// DepTypeDirect represents a direct dependency
	DepTypeDirect = "direct dep"

	// DepTypeTransitive represents a transitive dependency,
	// or a dependency of a dependency
	DepTypeTransitive = "transitive dep"

	// DepTypeImported represents a dependency imported by an external tool
	DepTypeImported = "imported dep"
)

// ConstraintFeedback holds project constraint feedback data
type ConstraintFeedback struct {
	Constraint, LockedVersion, Revision, ConstraintType, DependencyType, ProjectPath string
}

// NewConstraintFeedback builds a feedback entry for a constraint in the manifest.
func NewConstraintFeedback(pc gps.ProjectConstraint, depType string) *ConstraintFeedback {
	cf := &ConstraintFeedback{
		Constraint:     pc.Constraint.String(),
		ProjectPath:    string(pc.Ident.ProjectRoot),
		DependencyType: depType,
	}

	if _, ok := pc.Constraint.(gps.Revision); ok {
		cf.ConstraintType = ConsTypeHint
	} else {
		cf.ConstraintType = ConsTypeConstraint
	}

	return cf
}

// NewLockedProjectFeedback builds a feedback entry for a project in the lock.
func NewLockedProjectFeedback(lp gps.LockedProject, depType string) *ConstraintFeedback {
	cf := &ConstraintFeedback{
		ProjectPath:    string(lp.Ident().ProjectRoot),
		DependencyType: depType,
	}

	switch vt := lp.Version().(type) {
	case gps.PairedVersion:
		cf.LockedVersion = vt.String()
		cf.Revision = vt.Revision().String()
	case gps.UnpairedVersion: // Logically this should never occur, but handle for completeness sake
		cf.LockedVersion = vt.String()
	case gps.Revision:
		cf.Revision = vt.String()
	}

	return cf
}

// LogFeedback logs feedback on changes made to the manifest or lock.
func (cf ConstraintFeedback) LogFeedback(logger *log.Logger) {
	if cf.Constraint != "" {
		logger.Printf("  %v", GetUsingFeedback(cf.Constraint, cf.ConstraintType, cf.DependencyType, cf.ProjectPath))
	}
	if cf.Revision != "" {
		logger.Printf("  %v", GetLockingFeedback(cf.LockedVersion, cf.Revision, cf.DependencyType, cf.ProjectPath))
	}
}

// GetUsingFeedback returns a dependency "using" feedback message. For example:
//
//    Using ^1.0.0 as constraint for direct dep github.com/foo/bar
//    Using 1b8edb3 as hint for direct dep github.com/bar/baz
func GetUsingFeedback(version, consType, depType, projectPath string) string {
	if depType == DepTypeImported {
		return fmt.Sprintf("Using %s as initial %s for %s %s", version, consType, depType, projectPath)
	}
	return fmt.Sprintf("Using %s as %s for %s %s", version, consType, depType, projectPath)
}

// GetLockingFeedback returns a dependency "locking" feedback message. For
// example:
//
//    Locking in v1.1.4 (bc29b4f) for direct dep github.com/foo/bar
//    Locking in master (436f39d) for transitive dep github.com/baz/qux
func GetLockingFeedback(version, revision, depType, projectPath string) string {
	// Check if it's a valid SHA1 digest and trim to 7 characters.
	if len(revision) == 40 {
		if _, err := hex.DecodeString(revision); err == nil {
			// Valid SHA1 digest
			revision = revision[0:7]
		}
	}

	if depType == DepTypeImported {
		if version == "" {
			version = "*"
		}
		return fmt.Sprintf("Trying %s (%s) as initial lock for %s %s", version, revision, depType, projectPath)
	}
	return fmt.Sprintf("Locking in %s (%s) for %s %s", version, revision, depType, projectPath)
}
