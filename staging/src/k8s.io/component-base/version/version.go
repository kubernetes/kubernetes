/*
Copyright 2019 The Kubernetes Authors.

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

package version

import (
	"fmt"
	"runtime"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

var minimumKubeEmulationVersion *version.Version = version.MajorMinor(1, 31)

type EffectiveVersion interface {
	BinaryVersion() *version.Version
	EmulationVersion() *version.Version
	MinCompatibilityVersion() *version.Version
	EqualTo(other EffectiveVersion) bool
	String() string
	Validate() []error
}

type MutableEffectiveVersion interface {
	EffectiveVersion
	Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version)
	SetEmulationVersion(emulationVersion *version.Version)
	SetMinCompatibilityVersion(minCompatibilityVersion *version.Version)
}

type effectiveVersion struct {
	// When true, BinaryVersion() returns the current binary version
	useDefaultBuildBinaryVersion atomic.Bool
	// Holds the last binary version stored in Set()
	binaryVersion atomic.Pointer[version.Version]
	// If the emulationVersion is set by the users, it could only contain major and minor versions.
	// In tests, emulationVersion could be the same as the binary version, or set directly,
	// which can have "alpha" as pre-release to continue serving expired apis while we clean up the test.
	emulationVersion atomic.Pointer[version.Version]
	// minCompatibilityVersion could only contain major and minor versions.
	minCompatibilityVersion atomic.Pointer[version.Version]
}

// Get returns the overall codebase version. It's for detecting
// what code a binary was built from.
func Get() apimachineryversion.Info {
	// These variables typically come from -ldflags settings and in
	// their absence fallback to the settings in ./base.go
	return apimachineryversion.Info{
		Major:        gitMajor,
		Minor:        gitMinor,
		GitVersion:   dynamicGitVersion.Load().(string),
		GitCommit:    gitCommit,
		GitTreeState: gitTreeState,
		BuildDate:    buildDate,
		GoVersion:    runtime.Version(),
		Compiler:     runtime.Compiler,
		Platform:     fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}

func (m *effectiveVersion) BinaryVersion() *version.Version {
	if m.useDefaultBuildBinaryVersion.Load() {
		return defaultBuildBinaryVersion()
	}
	return m.binaryVersion.Load()
}

func (m *effectiveVersion) EmulationVersion() *version.Version {
	ver := m.emulationVersion.Load()
	if ver != nil {
		// Emulation version can have "alpha" as pre-release to continue serving expired apis while we clean up the test.
		// The pre-release should not be accessible to the users.
		return ver.WithPreRelease(m.BinaryVersion().PreRelease())
	}
	return ver
}

func (m *effectiveVersion) MinCompatibilityVersion() *version.Version {
	return m.minCompatibilityVersion.Load()
}

func (m *effectiveVersion) EqualTo(other EffectiveVersion) bool {
	return m.BinaryVersion().EqualTo(other.BinaryVersion()) && m.EmulationVersion().EqualTo(other.EmulationVersion()) && m.MinCompatibilityVersion().EqualTo(other.MinCompatibilityVersion())
}

func (m *effectiveVersion) String() string {
	if m == nil {
		return "<nil>"
	}
	return fmt.Sprintf("{BinaryVersion: %s, EmulationVersion: %s, MinCompatibilityVersion: %s}",
		m.BinaryVersion().String(), m.EmulationVersion().String(), m.MinCompatibilityVersion().String())
}

func majorMinor(ver *version.Version) *version.Version {
	if ver == nil {
		return ver
	}
	return version.MajorMinor(ver.Major(), ver.Minor())
}

func (m *effectiveVersion) Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version) {
	m.binaryVersion.Store(binaryVersion)
	m.useDefaultBuildBinaryVersion.Store(false)
	m.emulationVersion.Store(majorMinor(emulationVersion))
	m.minCompatibilityVersion.Store(majorMinor(minCompatibilityVersion))
}

func (m *effectiveVersion) SetEmulationVersion(emulationVersion *version.Version) {
	m.emulationVersion.Store(majorMinor(emulationVersion))
	// set the default minCompatibilityVersion to be emulationVersion - 1
	m.minCompatibilityVersion.Store(majorMinor(emulationVersion.SubtractMinor(1)))
}

// SetMinCompatibilityVersion should be called after SetEmulationVersion
func (m *effectiveVersion) SetMinCompatibilityVersion(minCompatibilityVersion *version.Version) {
	m.minCompatibilityVersion.Store(majorMinor(minCompatibilityVersion))
}

func (m *effectiveVersion) Validate() []error {
	var errs []error
	// Validate only checks the major and minor versions.
	binaryVersion := m.BinaryVersion().WithPatch(0)
	emulationVersion := m.emulationVersion.Load()
	minCompatibilityVersion := m.minCompatibilityVersion.Load()

	// emulationVersion can only be 1.{binaryMinor-3}...1.{binaryMinor}
	maxEmuVer := binaryVersion
	minEmuVer := binaryVersion.SubtractMinor(3)
	if emulationVersion.GreaterThan(maxEmuVer) || emulationVersion.LessThan(minEmuVer) {
		errs = append(errs, fmt.Errorf("emulation version %s is not between [%s, %s]", emulationVersion.String(), minEmuVer.String(), maxEmuVer.String()))
	}
	// minCompatibilityVersion can only be 1.{binaryMinor-3} to 1.{binaryMinor}
	maxCompVer := emulationVersion
	minCompVer := binaryVersion.SubtractMinor(4)
	if minCompatibilityVersion.GreaterThan(maxCompVer) || minCompatibilityVersion.LessThan(minCompVer) {
		errs = append(errs, fmt.Errorf("minCompatibilityVersion version %s is not between [%s, %s]", minCompatibilityVersion.String(), minCompVer.String(), maxCompVer.String()))
	}
	return errs
}

func newEffectiveVersion(binaryVersion *version.Version, useDefaultBuildBinaryVersion bool) MutableEffectiveVersion {
	effective := &effectiveVersion{}
	compatVersion := binaryVersion.SubtractMinor(1)
	effective.Set(binaryVersion, binaryVersion, compatVersion)
	effective.useDefaultBuildBinaryVersion.Store(useDefaultBuildBinaryVersion)
	return effective
}

func NewEffectiveVersion(binaryVer string) MutableEffectiveVersion {
	if binaryVer == "" {
		return &effectiveVersion{}
	}
	binaryVersion := version.MustParse(binaryVer)
	return newEffectiveVersion(binaryVersion, false)
}

func defaultBuildBinaryVersion() *version.Version {
	verInfo := Get()
	return version.MustParse(verInfo.String()).WithInfo(verInfo)
}

// DefaultBuildEffectiveVersion returns the MutableEffectiveVersion based on the
// current build information.
func DefaultBuildEffectiveVersion() MutableEffectiveVersion {
	binaryVersion := defaultBuildBinaryVersion()
	if binaryVersion.Major() == 0 && binaryVersion.Minor() == 0 {
		return DefaultKubeEffectiveVersion()
	}
	return newEffectiveVersion(binaryVersion, true)
}

// DefaultKubeEffectiveVersion returns the MutableEffectiveVersion based on the
// latest K8s release.
func DefaultKubeEffectiveVersion() MutableEffectiveVersion {
	binaryVersion := version.MustParse(DefaultKubeBinaryVersion).WithInfo(Get())
	return newEffectiveVersion(binaryVersion, false)
}

// ValidateKubeEffectiveVersion validates the EmulationVersion is at least 1.31 and MinCompatibilityVersion
// is at least 1.30 for kube components.
func ValidateKubeEffectiveVersion(effectiveVersion EffectiveVersion) error {
	if !effectiveVersion.EmulationVersion().AtLeast(minimumKubeEmulationVersion) {
		return fmt.Errorf("emulation version needs to be greater or equal to 1.31, got %s", effectiveVersion.EmulationVersion().String())
	}
	if !effectiveVersion.MinCompatibilityVersion().AtLeast(minimumKubeEmulationVersion.SubtractMinor(1)) {
		return fmt.Errorf("minCompatibilityVersion version needs to be greater or equal to 1.30, got %s", effectiveVersion.MinCompatibilityVersion().String())
	}

	return nil
}
