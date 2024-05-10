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

package version

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	baseversion "k8s.io/component-base/version"
)

var DefaultEffectiveVersionRegistry EffectiveVersionRegistry = NewEffectiveVersionRegistry()

const (
	ComponentGenericAPIServer = "k8s.io/apiserver"
)

type EffectiveVersionRegistry interface {
	// EffectiveVersionFor returns the EffectiveVersion registered under the component.
	// Returns nil if the component is not registered.
	EffectiveVersionFor(component string) MutableEffectiveVersion
	// EffectiveVersionForOrRegister returns the EffectiveVersion registered under the component if it is already registered
	// (ignoring any conflict between the registered version and the provided version).
	// If the component is not registered, it would register the provided ver under the component, and return the same ver.
	EffectiveVersionForOrRegister(component string, ver MutableEffectiveVersion) MutableEffectiveVersion
	// RegisterEffectiveVersionFor registers the EffectiveVersion for a component.
	// Overrides existing EffectiveVersion if it is already in the registry.
	RegisterEffectiveVersionFor(component string, ver MutableEffectiveVersion)
}

type effectiveVersionRegistry struct {
	effectiveVersions map[string]MutableEffectiveVersion
	mutex             sync.RWMutex
}

func NewEffectiveVersionRegistry() EffectiveVersionRegistry {
	return &effectiveVersionRegistry{effectiveVersions: map[string]MutableEffectiveVersion{}}
}

func (r *effectiveVersionRegistry) EffectiveVersionFor(component string) MutableEffectiveVersion {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	return r.effectiveVersions[component]
}

func (r *effectiveVersionRegistry) EffectiveVersionForOrRegister(component string, ver MutableEffectiveVersion) MutableEffectiveVersion {
	r.mutex.RLock()
	v, ok := r.effectiveVersions[component]
	r.mutex.RUnlock()
	if ok {
		return v
	}
	r.RegisterEffectiveVersionFor(component, ver)
	return ver
}

func (r *effectiveVersionRegistry) RegisterEffectiveVersionFor(component string, ver MutableEffectiveVersion) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.effectiveVersions[component] = ver
}

type EffectiveVersion interface {
	BinaryVersion() *version.Version
	EmulationVersion() *version.Version
	MinCompatibilityVersion() *version.Version
	// VersionInfo() stores full info about the binary version. Mainly used for /version endpoint.
	VersionInfo() *apimachineryversion.Info
	EqualTo(other EffectiveVersion) bool
	String() string
	Validate() []error
}

type MutableEffectiveVersion interface {
	EffectiveVersion
	Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version)
	// AddFlags adds the "{prefix}-emulated-version" to the flagset.
	AddFlags(fs *pflag.FlagSet, prefix string)
	SetVersionInfo(versionInfo *apimachineryversion.Info)
}

type VersionVar struct {
	val atomic.Pointer[version.Version]
}

// Set sets the flag value
func (v *VersionVar) Set(s string) error {
	components := strings.Split(s, ".")
	if len(components) != 2 {
		return fmt.Errorf("version %s is not in the format of major.minor", s)
	}
	ver, err := version.ParseGeneric(s)
	if err != nil {
		return err
	}
	v.val.Store(ver)
	return nil
}

// String returns the flag value
func (v *VersionVar) String() string {
	ver := v.val.Load()
	return ver.String()
}

// Type gets the flag type
func (v *VersionVar) Type() string {
	return "version"
}

type effectiveVersion struct {
	binaryVersion atomic.Pointer[version.Version]
	// If the emulationVersion is set by the users, it could only contain major and minor versions.
	// In tests, emulationVersion could be the same as the binary version, or set directly,
	// which can have "alpha" as pre-release to continue serving expired apis while we clean up the test.
	emulationVersion VersionVar
	// minCompatibilityVersion could only contain major and minor versions.
	minCompatibilityVersion VersionVar
	versionInfo             atomic.Pointer[apimachineryversion.Info]
}

func (m *effectiveVersion) BinaryVersion() *version.Version {
	return m.binaryVersion.Load()
}

func (m *effectiveVersion) EmulationVersion() *version.Version {
	// Emulation version can have "alpha" as pre-release to continue serving expired apis while we clean up the test.
	// The pre-release should not be accessible to the users.
	return m.emulationVersion.val.Load().WithPreRelease(m.BinaryVersion().PreRelease())
}

func (m *effectiveVersion) MinCompatibilityVersion() *version.Version {
	return m.minCompatibilityVersion.val.Load()
}

func (m *effectiveVersion) VersionInfo() *apimachineryversion.Info {
	return m.versionInfo.Load()
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

func (m *effectiveVersion) Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version) {
	m.binaryVersion.Store(binaryVersion)
	m.emulationVersion.val.Store(version.MajorMinor(emulationVersion.Major(), emulationVersion.Minor()))
	m.minCompatibilityVersion.val.Store(version.MajorMinor(minCompatibilityVersion.Major(), minCompatibilityVersion.Minor()))
}

func (m *effectiveVersion) Validate() []error {
	var errs []error
	// Validate only checks the major and minor versions.
	binaryVersion := m.binaryVersion.Load().WithPatch(0)
	emulationVersion := m.emulationVersion.val.Load()
	minCompatibilityVersion := m.minCompatibilityVersion.val.Load()

	// emulationVersion can only be 1.{binaryMinor-1}...1.{binaryMinor}.
	maxEmuVer := binaryVersion
	minEmuVer := binaryVersion.SubtractMinor(1)
	// TODO: remove in 1.32
	// emulationVersion is introduced in 1.31, so it cannot be lower than that.
	// binaryVersion could be lower than 1.31 in tests. So we are only checking 1.31.
	if binaryVersion.EqualTo(version.MajorMinor(1, 31)) {
		minEmuVer = version.MajorMinor(1, 31)
	}
	if emulationVersion.GreaterThan(maxEmuVer) || emulationVersion.LessThan(minEmuVer) {
		errs = append(errs, fmt.Errorf("emulation version %s is not between [%s, %s]", emulationVersion.String(), minEmuVer.String(), maxEmuVer.String()))
	}
	// minCompatibilityVersion can only be 1.{binaryMinor-1} for alpha.
	maxCompVer := binaryVersion.SubtractMinor(1)
	minCompVer := binaryVersion.SubtractMinor(1)
	if minCompatibilityVersion.GreaterThan(maxCompVer) || minCompatibilityVersion.LessThan(minCompVer) {
		errs = append(errs, fmt.Errorf("minCompatibilityVersion version %s is not between [%s, %s]", minCompatibilityVersion.String(), minCompVer.String(), maxCompVer.String()))
	}
	return errs
}

// AddFlags adds the "{prefix}-emulated-version" to the flagset.
func (m *effectiveVersion) AddFlags(fs *pflag.FlagSet, prefix string) {
	if m == nil {
		return
	}
	if len(prefix) > 0 && !strings.HasSuffix(prefix, "-") {
		prefix += "-"
	}
	fs.Var(&m.emulationVersion, prefix+"emulated-version", ""+
		"The version the K8s component emulates its capabilities (APIs, features, ...) of.\n"+
		"If set, the component will emulate the behavior of this version instead of the underlying binary version.\n"+
		"Any capabilities present in the binary version that were introduced after the emulated version will be unavailable and any capabilities removed after the emulated version will be available.\n"+
		"This flag applies only to component capabilities, and does not disable bug fixes and performance improvements present in the binary version.\n"+
		"Defaults to the binary version. The value should be between 1.{binaryMinorVersion-1} and 1.{binaryMinorVersion}.\n"+
		"Format could only be major.minor")
}

func (m *effectiveVersion) SetVersionInfo(versionInfo *apimachineryversion.Info) {
	m.versionInfo.Store(versionInfo)
}

func NewEffectiveVersion(binaryVer string) MutableEffectiveVersion {
	effective := &effectiveVersion{}
	binaryVersion := version.MustParse(binaryVer)
	compatVersion := binaryVersion.SubtractMinor(1)
	effective.Set(binaryVersion, binaryVersion, compatVersion)
	effective.SetVersionInfo(binaryVersion.VersionInfo())
	return effective
}

// DefaultBuildEffectiveVersion returns the MutableEffectiveVersion based on the
// current build information.
func DefaultBuildEffectiveVersion() MutableEffectiveVersion {
	verInfo := baseversion.Get()
	ver := NewEffectiveVersion(verInfo.String())
	if ver.BinaryVersion().Major() == 0 && ver.BinaryVersion().Minor() == 0 {
		ver = DefaultKubeEffectiveVersion()
	}
	ver.SetVersionInfo(&verInfo)
	return ver
}

// DefaultKubeEffectiveVersion returns the MutableEffectiveVersion based on the
// latest K8s release.
// Should update for each minor release!
func DefaultKubeEffectiveVersion() MutableEffectiveVersion {
	return NewEffectiveVersion("1.31")
}
