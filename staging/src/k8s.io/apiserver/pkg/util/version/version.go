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
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/version"
	baseversion "k8s.io/component-base/version"
)

var Effective MutableEffectiveVersions = newEffectiveVersion()

type EffectiveVersions interface {
	BinaryVersion() *version.Version
	EmulationVersion() *version.Version
	MinCompatibilityVersion() *version.Version
}

type MutableEffectiveVersions interface {
	EffectiveVersions
	Set(binaryVersion, emulationVersion, minCompatibilityVersion *version.Version)
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
	binaryVersion := version.MustParseGeneric(binaryVersionInfo.String())
	compatVersion := version.MajorMinor(binaryVersion.Major(), binaryVersion.Minor()-1)
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

func (m *mutableEffectiveVersions) SetEmulationVersion(emulationVersion *version.Version) {
	m.Set(m.BinaryVersion(), emulationVersion, m.MinCompatibilityVersion())
}

func (m *mutableEffectiveVersions) SetMinCompatibilityVersion(minCompatibilityVersion *version.Version) {
	m.Set(m.BinaryVersion(), m.EmulationVersion(), minCompatibilityVersion)
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
