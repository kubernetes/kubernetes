/*
   Copyright © The CDI Authors

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

package specs

import (
	"fmt"
	"strings"
)

// CurrentVersion is the current version of the Spec.
const CurrentVersion = "1.1.0"

const (
	// These represent the released versions of the CDI specification
	v010 version = "0.1.0"
	v020 version = "0.2.0"
	v030 version = "0.3.0"
	v040 version = "0.4.0"
	v050 version = "0.5.0"
	v060 version = "0.6.0"
	v070 version = "0.7.0"
	v080 version = "0.8.0"
	v100 version = "1.0.0"
	v110 version = "1.1.0"

	// vEarliest is the earliest supported version of the CDI specification
	vEarliest version = v030
)

// validSpecVersions stores the known spec versions in newest-to-oldest order,
// together with functions to check whether a version is required.
// Adding new fields / spec versions requires that a `requiredFunc` be implemented and
// this list be updated.
var validSpecVersions = []struct {
	version    version
	isRequired requiredFunc
}{
	{v110, requiresV110},
	{v100, requiresV100},
	{v080, requiresV080},
	{v070, requiresV070},
	{v060, requiresV060},
	{v050, requiresV050},
	{v040, requiresV040},
	{v030, nil},
	{v020, nil},
	{v010, nil},
}

// ValidateVersion checks whether the specified spec version is valid.
// In addition to checking whether the spec version is in the set of known versions,
// the spec is inspected to determine whether the features used are available in specified
// version.
func ValidateVersion(spec *Spec) error {
	specVersion := newVersion(spec.Version)
	if !isValidVersion(specVersion) {
		return fmt.Errorf("invalid version %q", spec.Version)
	}
	minVersion, err := MinimumRequiredVersion(spec)
	if err != nil {
		return fmt.Errorf("could not determine minimum required version: %w", err)
	}
	if versionIndex(version(minVersion)) < versionIndex(specVersion) {
		return fmt.Errorf("the spec version must be at least v%v", minVersion)
	}
	return nil
}

// MinimumRequiredVersion determines the minimum spec version for the input spec.
func MinimumRequiredVersion(spec *Spec) (string, error) {
	minVersion := requiredVersion(spec)
	return string(minVersion), nil
}

// version represents a CDI specification version.
type version string

// newVersion normalizes a specification version by removing an optional leading v.
func newVersion(v string) version {
	return version(strings.TrimPrefix(v, "v"))
}

type requiredFunc func(*Spec) bool

// isValidVersion checks whether the specified version is valid.
// A version is valid if it is contained in the list of known spec versions.
func isValidVersion(specVersion version) bool {
	for _, known := range validSpecVersions {
		if known.version == specVersion {
			return true
		}
	}
	return false
}

// requiredVersion returns the minimum version required for the given spec.
func requiredVersion(spec *Spec) version {
	for _, known := range validSpecVersions {
		if known.isRequired != nil && known.isRequired(spec) {
			return known.version
		}
	}
	return vEarliest
}

// versionIndex returns the index of v in validSpecVersions, which is ordered
// newest-to-oldest. It returns -1 for an unknown version.
func versionIndex(v version) int {
	for i, known := range validSpecVersions {
		if known.version == v {
			return i
		}
	}
	return -1
}

// requiresV110 returns true if the spec uses v1.1.0 features.
func requiresV110(spec *Spec) bool {
	if i := spec.ContainerEdits.IntelRdt; i != nil {
		if i.Schemata != nil || i.EnableMonitoring {
			return true
		}
	}

	if len(spec.ContainerEdits.NetDevices) > 0 {
		return true
	}

	for _, dev := range spec.Devices {
		if i := dev.ContainerEdits.IntelRdt; i != nil {
			if i.Schemata != nil || i.EnableMonitoring {
				return true
			}
		}

		if len(dev.ContainerEdits.NetDevices) > 0 {
			return true
		}
	}

	return false
}

// requiresV100 returns true if the spec uses v1.0.0 features.
// Since the v1.0.0 spec bump was due to moving the minimum version checks to
// the spec package, there are no explicit spec changes.
func requiresV100(_ *Spec) bool {
	return false
}

// requiresV080 returns true if the spec uses v0.8.0 features.
// Since the v0.8.0 spec bump was due to the removed .ToOCI functions on the
// spec types, there are no explicit spec changes.
func requiresV080(_ *Spec) bool {
	return false
}

// requiresV070 returns true if the spec uses v0.7.0 features
func requiresV070(spec *Spec) bool {
	if spec.ContainerEdits.IntelRdt != nil {
		return true
	}
	// The v0.7.0 spec allows additional GIDs to be specified at a spec level.
	if len(spec.ContainerEdits.AdditionalGIDs) > 0 {
		return true
	}

	for _, d := range spec.Devices {
		if d.ContainerEdits.IntelRdt != nil {
			return true
		}
		// The v0.7.0 spec allows additional GIDs to be specified at a device level.
		if len(d.ContainerEdits.AdditionalGIDs) > 0 {
			return true
		}
	}

	return false
}

// requiresV060 returns true if the spec uses v0.6.0 features
func requiresV060(spec *Spec) bool {
	// The v0.6.0 spec allows annotations to be specified at a spec level
	if len(spec.Annotations) > 0 {
		return true
	}

	// The v0.6.0 spec allows annotations to be specified at a device level
	for _, d := range spec.Devices {
		if len(d.Annotations) > 0 {
			return true
		}
	}

	// The v0.6.0 spec allows dots "." in Kind name label (class)
	_, class, ok := strings.Cut(spec.Kind, "/")
	return ok && strings.Contains(class, ".")
}

// requiresV050 returns true if the spec uses v0.5.0 features
func requiresV050(spec *Spec) bool {
	var edits []*ContainerEdits

	for _, d := range spec.Devices {
		// The v0.5.0 spec allowed device name to start with a digit
		if len(d.Name) > 0 && '0' <= d.Name[0] && d.Name[0] <= '9' {
			return true
		}
		edits = append(edits, &d.ContainerEdits)
	}

	edits = append(edits, &spec.ContainerEdits)
	for _, e := range edits {
		for _, dn := range e.DeviceNodes {
			// The HostPath field was added in v0.5.0
			if dn.HostPath != "" {
				return true
			}
		}
	}
	return false
}

// requiresV040 returns true if the spec uses v0.4.0 features
func requiresV040(spec *Spec) bool {
	var edits []*ContainerEdits

	for _, d := range spec.Devices {
		edits = append(edits, &d.ContainerEdits)
	}

	edits = append(edits, &spec.ContainerEdits)
	for _, e := range edits {
		for _, m := range e.Mounts {
			// The Type field was added in v0.4.0
			if m.Type != "" {
				return true
			}
		}
	}
	return false
}
