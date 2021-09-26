// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package modinfo

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/delta"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/report"
	"github.com/Azure/azure-sdk-for-go/tools/internal/dirs"
	"github.com/Masterminds/semver"
)

var (
	verSuffixRegex = regexp.MustCompile(`v\d+$`)
)

// HasVersionSuffix returns true if the specified path has a version suffix in the form vN.
func HasVersionSuffix(path string) bool {
	return verSuffixRegex.MatchString(path)
}

// FindVersionSuffix returns the version suffix or the empty string.
func FindVersionSuffix(path string) string {
	return verSuffixRegex.FindString(path)
}

// GetModuleSubdirs returns all subdirectories under path that correspond to module major versions.
// The subdirectories are sorted according to semantic version.
func GetModuleSubdirs(path string) ([]string, error) {
	subdirs, err := dirs.GetSubdirs(path)
	if err != nil {
		return nil, err
	}
	modDirs := []string{}
	for _, subdir := range subdirs {
		matched := HasVersionSuffix(subdir)
		if matched {
			modDirs = append(modDirs, subdir)
		}
	}
	sortModuleTagsBySemver(modDirs)
	return modDirs, nil
}

// IncrementModuleVersion increments the passed in module major version by one.
// E.g. a provided value of "v2" will return "v3".  If ver is "" the return value is "v2".
func IncrementModuleVersion(ver string) string {
	if ver == "" {
		return "v2"
	}
	v, err := strconv.ParseInt(ver[1:], 10, 32)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("v%d", v+1)
}

// sorts module tags based on their semantic versions.
// this is necessary because lexically sorted is not sufficient
// due to v10.0.0 appearing before v2.0.0
func sortModuleTagsBySemver(modDirs []string) {
	sort.SliceStable(modDirs, func(i, j int) bool {
		lv, err := semver.NewVersion(modDirs[i])
		if err != nil {
			panic(err)
		}
		rv, err := semver.NewVersion(modDirs[j])
		if err != nil {
			panic(err)
		}
		return lv.LessThan(rv)
	})
}

// CreateModuleNameFromPath creates a module name from the provided path.
func CreateModuleNameFromPath(pkgDir string) (string, error) {
	// e.g. /work/src/github.com/Azure/azure-sdk-for-go/services/foo/2019-01-01/foo
	// returns github.com/Azure/azure-sdk-for-go/services/foo/2019-01-01/foo
	repoRoot := filepath.Join("github.com", "Azure", "azure-sdk-for-go")
	i := strings.Index(pkgDir, repoRoot)
	if i < 0 {
		return "", fmt.Errorf("didn't find '%s' in '%s'", repoRoot, pkgDir)
	}
	return strings.Replace(pkgDir[i:], "\\", "/", -1), nil
}

// Provider provides information about a module staged for release.
type Provider interface {
	DestDir() string
	NewExports() bool
	BreakingChanges() bool
	VersionSuffix() bool
	NewModule() bool
	GenerateReport() report.Package
}

type module struct {
	lhs  exports.Content
	rhs  exports.Content
	dest string
}

// GetModuleInfo collects information about a module staged for release.
// baseline is the directory for the current module
// staged is the directory for the module staged for release
func GetModuleInfo(baseline, staged string) (Provider, error) {
	// TODO: verify staged is a child of baseline
	lhs, err := exports.Get(baseline)
	if err != nil {
		// if baseline has no content then this is a v1 package
		if ei, ok := err.(exports.LoadPackageErrorInfo); !ok || ei.PkgCount() != 0 {
			return nil, fmt.Errorf("failed to get exports for package '%s': %s", baseline, err)
		}
	}
	rhs, err := exports.Get(staged)
	if err != nil {
		return nil, fmt.Errorf("failed to get exports for package '%s': %s", staged, err)
	}
	mod := module{
		lhs:  lhs,
		rhs:  rhs,
		dest: baseline,
	}
	// calculate the destination directory
	// if there are breaking changes calculate the new directory
	if mod.BreakingChanges() {
		dest := filepath.Dir(staged)
		v := 2
		if verSuffixRegex.MatchString(baseline) {
			// baseline has a version, get the number and increment it
			s := string(baseline[len(baseline)-1])
			v, err = strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("failed to convert '%s' to int: %v", s, err)
			}
			v++
		}
		mod.dest = filepath.Join(dest, fmt.Sprintf("v%d", v))
	}
	return mod, nil
}

// DestDir returns the fully qualified module destination directory.
func (m module) DestDir() string {
	return m.dest
}

// NewExports returns true if the module contains any additive changes.
func (m module) NewExports() bool {
	if m.lhs.IsEmpty() {
		return true
	}
	adds := delta.GetExports(m.lhs, m.rhs)
	return !adds.IsEmpty()
}

// BreakingChanges returns true if the module contains breaking changes.
func (m module) BreakingChanges() bool {
	if m.lhs.IsEmpty() {
		return false
	}
	// check for changed content
	if len(delta.GetConstTypeChanges(m.lhs, m.rhs)) > 0 ||
		len(delta.GetFuncSigChanges(m.lhs, m.rhs)) > 0 ||
		len(delta.GetInterfaceMethodSigChanges(m.lhs, m.rhs)) > 0 ||
		len(delta.GetStructFieldChanges(m.lhs, m.rhs)) > 0 {
		return true
	}
	// check for removed content
	if removed := delta.GetExports(m.rhs, m.lhs); !removed.IsEmpty() {
		return true
	}
	return false
}

// VersionSuffix returns true if the module path contains a version suffix.
func (m module) VersionSuffix() bool {
	return verSuffixRegex.MatchString(m.dest)
}

// NewModule returns true if the module is new, i.e. v1.0.0.
func (m module) NewModule() bool {
	return m.lhs.IsEmpty()
}

// GenerateReport generates a package report for the module.
func (m module) GenerateReport() report.Package {
	return report.Generate(m.lhs, m.rhs, false, false)
}

// IsValidModuleVersion returns true if the provided string is a valid module version (e.g. v1.2.3).
func IsValidModuleVersion(v string) bool {
	r := regexp.MustCompile(`^v\d+\.\d+\.\d+$`)
	return r.MatchString(v)
}
