/*
Copyright The Kubernetes Authors.

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

package apidefinitions

import (
	"fmt"
	"strings"

	"k8s.io/gengo/v2/types"
)

// Activation represents the data extracted from an API package's
// doc.go that control the activation of a generator.
type Activation struct {
	// spec is the generator specification this activation applies to.
	spec Spec
	// shouldGenerate reports if the spec generator should be run
	// for the package.
	shouldGenerate bool

	// typeFilers is only set if the spec's ValueMode is TypeFilterList
	typeFilters []string

	// externalTypes provides a full import path to the external types.
	externalTypes string

	// Conversion specific:
	// explicitOnly indicates that conversion is configured to only
	// generate conversion for explicitly tagged types.
	explicitOnly bool
	// peerPackages provides the conversion peer package list.
	peerPackages []string
}

// ShouldGenerate reports whether the generator should run on the
// package. It unifies all activation paths: explicit ActivationTag
// value, =false opt-out, +<GroupActivationTag> fallback, and ScanTypes
// type-level fallback. Generators with ValueMode ConversionPeerList
// also generate when ActivationTag is "false" (in explicit-only mode);
// use IsExplicitOnly to disambiguate.
func (a *Activation) ShouldGenerate() bool { return a.shouldGenerate }

// TypeFilters returns the type name patterns the generator should
// process.
func (a *Activation) TypeFilters() []string { return a.typeFilters }

// ExternalTypes returns a full import path to the external types.
func (a *Activation) ExternalTypes() string { return a.externalTypes }

// PeerPackages returns the conversion peer package list.
func (a *Activation) PeerPackages() []string { return a.peerPackages }

// IsExplicitOnly reports whether conversion-gen should run in
// explicit-only mode (emit conversions only for explicitly tagged
// types). Set when +k8s:conversion-gen=false; ONLY meaningful for
// generators using ConversionPeerList.
func (a *Activation) IsExplicitOnly() bool { return a.explicitOnly }

// Identify extracts data from pkg's doc.go for the generator described
// by spec.
func Identify(pkg *types.Package, spec Spec, opts ...Option) (*Activation, error) {
	var st identifyState
	for _, opt := range opts {
		opt(&st)
	}
	if err := applyLintRules(&st, pkg); err != nil {
		return nil, err
	}
	a := &Activation{spec: spec}
	if pkg == nil {
		return a, nil
	}
	values, optedOut, err := extractGeneratorTag(pkg.Comments, spec.ActivationTag)
	if err != nil {
		return nil, err
	}
	enabled := !optedOut && len(values) > 0
	switch spec.ValueMode {
	case ConversionPeerList:
		for _, v := range values {
			if err := checkRelativeImportPath(spec.ActivationTag, v); err != nil {
				return nil, err
			}
		}
		a.peerPackages = values
		a.explicitOnly = optedOut
	case TypeFilterList:
		a.typeFilters = values
	}
	a.externalTypes, err = resolveExternalTypes(pkg, spec)
	if err != nil {
		return nil, err
	}

	switch {
	case enabled:
		a.shouldGenerate = true
	case optedOut:
		// Only for ConversionPeerList can is generation still needed.
		// This is to handle the explicit-only mode of conversion-gen.
		a.shouldGenerate = spec.ValueMode == ConversionPeerList
	case spec.DefaultEnabled:
		a.shouldGenerate = true
	case isGroupActivated(pkg, spec):
		a.shouldGenerate = true
	case spec.ScanTypes:
		a.shouldGenerate = scanTypesForActivation(pkg, spec.ActivationTag)
	}
	return a, nil
}

func isGroupActivated(pkg *types.Package, spec Spec) bool {
	if len(spec.GroupActivationTag) > 0 {
		if vals, _ := tagValues(pkg.Comments, spec.GroupActivationTag); len(vals) > 0 {
			return true
		}
	}
	return false
}

// scanTypesForActivation reports whether any type in pkg has a
// type-level +<tagName>=<non-false> tag. This implements ScanTypes
// activation check.
func scanTypesForActivation(pkg *types.Package, tagName string) bool {
	if tagName == "" {
		return false
	}
	for _, t := range pkg.Types {
		lines := append(append([]string{}, t.CommentLines...), t.SecondClosestCommentLines...)
		vals, err := tagValues(lines, tagName)
		if err != nil {
			continue
		}
		for _, v := range vals {
			if v != "false" {
				return true
			}
		}
	}
	return false
}

// resolveExternalTypes returns the value of pkg's +<spec.InputTag> if
// present, or pkg.Path otherwise. Multiple values for the input tag is
// an error.
func resolveExternalTypes(pkg *types.Package, spec Spec) (string, error) {
	if spec.InputTag == "" {
		return pkg.Path, nil
	}
	values, err := tagValues(pkg.Comments, spec.InputTag)
	if err != nil {
		return "", err
	}
	switch len(values) {
	case 0:
		return pkg.Path, nil
	case 1:
		if err := checkRelativeImportPath(spec.InputTag, values[0]); err != nil {
			return "", err
		}
		return values[0], nil
	default:
		return "", fmt.Errorf("+%s: expected at most one value, got %v", spec.InputTag, values)
	}
}

// checkRelativeImportPath checks if the path is a relative path and returns an error if so
func checkRelativeImportPath(tagName, path string) error {
	if strings.HasPrefix(path, "./") || strings.HasPrefix(path, "../") {
		return fmt.Errorf("+%s: relative path %q is not supported; use the full package import path", tagName, path)
	}
	return nil
}
