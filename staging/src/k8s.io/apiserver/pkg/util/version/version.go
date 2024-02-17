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
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	baseversion "k8s.io/component-base/version"
)

var Effective MutableEffectiveVersions = newEffectiveVersion()

type EffectiveVersions interface {
	BinaryVersion() *version.Version
	EmulationVersion() *version.Version
	MinCompatibilityVersion() *version.Version
	Validate() []error
}

type MutableEffectiveVersions interface {
	EffectiveVersions
	// SetBinaryVersionForTests updates the binaryVersion.
	// Should only be used in tests.
	SetBinaryVersionForTests(binaryVersion *version.Version) func()
	SetEmulationVersion(emulationVersion *version.Version)
	SetMinCompatibilityVersion(minCompatibilityVersion *version.Version)
}

type effectiveVersions struct {
	binaryVersion           *version.Version
	emulationVersion        *version.Version
	minCompatibilityVersion *version.Version
}

func (m *effectiveVersions) BinaryVersion() *version.Version {
	return m.binaryVersion
}

func (m *effectiveVersions) EmulationVersion() *version.Version {
	return m.emulationVersion
}

func (m *effectiveVersions) MinCompatibilityVersion() *version.Version {
	return m.minCompatibilityVersion
}

type mutableEffectiveVersions struct {
	effectiveVersions atomic.Pointer[effectiveVersions]
}

func newEffectiveVersion() MutableEffectiveVersions {
	effective := &mutableEffectiveVersions{}
	binaryVersionInfo := baseversion.Get()
	binaryVersion := version.MustParse(binaryVersionInfo.String())
	compatVersion := version.MajorMinor(binaryVersion.Major(), binaryVersion.SubtractMinor(1).Minor())
	effective.Set(binaryVersion, binaryVersion, compatVersion)
	return effective
}

func (m *mutableEffectiveVersions) Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version) {
	m.effectiveVersions.Store(&effectiveVersions{
		binaryVersion:           binaryVersion,
		emulationVersion:        emulationVersion,
		minCompatibilityVersion: minCompatibilityVersion,
	})
}

// SetBinaryVersionForTests updates the binaryVersion.
// Should only be used in tests.
func (m *mutableEffectiveVersions) SetBinaryVersionForTests(binaryVersion *version.Version) func() {
	effectiveVersions := m.effectiveVersions.Load()
	m.Set(binaryVersion, binaryVersion, version.MajorMinor(binaryVersion.Major(), binaryVersion.Minor()-1))
	return func() {
		m.effectiveVersions.Store(effectiveVersions)
	}
}

func (m *mutableEffectiveVersions) SetEmulationVersion(emulationVersion *version.Version) {
	effectiveVersions := m.effectiveVersions.Load()
	m.Set(effectiveVersions.binaryVersion, emulationVersion, effectiveVersions.minCompatibilityVersion)
}

func (m *mutableEffectiveVersions) SetMinCompatibilityVersion(minCompatibilityVersion *version.Version) {
	effectiveVersions := m.effectiveVersions.Load()
	m.Set(effectiveVersions.binaryVersion, effectiveVersions.emulationVersion, minCompatibilityVersion)
}

func (m *mutableEffectiveVersions) BinaryVersion() *version.Version {
	return m.effectiveVersions.Load().BinaryVersion()
}

func (m *mutableEffectiveVersions) EmulationVersion() *version.Version {
	return m.effectiveVersions.Load().EmulationVersion()
}

func (m *mutableEffectiveVersions) MinCompatibilityVersion() *version.Version {
	return m.effectiveVersions.Load().MinCompatibilityVersion()
}

func (m *mutableEffectiveVersions) Validate() []error {
	ev := m.effectiveVersions.Load()
	// Validate only checks the major and minor versions.
	binaryVersion := ev.binaryVersion.WithPatch(0)
	emulationVersion := ev.emulationVersion.WithPatch(0)
	minCompatibilityVersion := ev.minCompatibilityVersion.WithPatch(0)

	maxEmuVer := binaryVersion
	minEmuVer := binaryVersion
	// emulationVersion can only be 1.{binaryMinor-1}...1.{binaryMinor} for alpha if EmulationVersion feature gate is enabled.
	// otherwise, emulationVersion can only be 1.{binaryMinor}.
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.EmulationVersion) {
		minEmuVer = binaryVersion.SubtractMinor(1)
	}
	if emulationVersion.GreaterThan(maxEmuVer) || emulationVersion.LessThan(minEmuVer) {
		return []error{fmt.Errorf("emulation version %s is not between [%s, %s]", emulationVersion.String(), minEmuVer.String(), maxEmuVer.String())}
	}
	// minCompatibilityVersion can only be 1.{binaryMinor-1} for alpha.
	maxCompVer := binaryVersion.SubtractMinor(1)
	minCompVer := binaryVersion.SubtractMinor(1)
	if minCompatibilityVersion.GreaterThan(maxCompVer) || minCompatibilityVersion.LessThan(minCompVer) {
		return []error{fmt.Errorf("minCompatibilityVersion version %s is not between [%s, %s]", minCompatibilityVersion.String(), minCompVer.String(), maxCompVer.String())}
	}
	return []error{}
}
