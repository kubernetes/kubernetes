/*
   Copyright © 2021 The CDI Authors

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

package cdi

import (
	"io/ioutil"
	"os"
	"path/filepath"

	oci "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
	"sigs.k8s.io/yaml"

	cdi "github.com/container-orchestrated-devices/container-device-interface/specs-go"
)

var (
	// Valid CDI Spec versions.
	validSpecVersions = map[string]struct{}{
		"0.1.0": {},
		"0.2.0": {},
		"0.3.0": {},
		"0.4.0": {},
	}

	// Externally set CDI Spec validation function.
	specValidator func(*cdi.Spec) error
)

// Spec represents a single CDI Spec. It is usually loaded from a
// file and stored in a cache. The Spec has an associated priority.
// This priority is inherited from the associated priority of the
// CDI Spec directory that contains the CDI Spec file and is used
// to resolve conflicts if multiple CDI Spec files contain entries
// for the same fully qualified device.
type Spec struct {
	*cdi.Spec
	vendor   string
	class    string
	path     string
	priority int
	devices  map[string]*Device
}

// ReadSpec reads the given CDI Spec file. The resulting Spec is
// assigned the given priority. If reading or parsing the Spec
// data fails ReadSpec returns a nil Spec and an error.
func ReadSpec(path string, priority int) (*Spec, error) {
	data, err := ioutil.ReadFile(path)
	switch {
	case os.IsNotExist(err):
		return nil, err
	case err != nil:
		return nil, errors.Wrapf(err, "failed to read CDI Spec %q", path)
	}

	raw, err := parseSpec(data)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse CDI Spec %q", path)
	}
	if raw == nil {
		return nil, errors.Errorf("failed to parse CDI Spec %q, no Spec data", path)
	}

	spec, err := NewSpec(raw, path, priority)
	if err != nil {
		return nil, err
	}

	return spec, nil
}

// NewSpec creates a new Spec from the given CDI Spec data. The
// Spec is marked as loaded from the given path with the given
// priority. If Spec data validation fails NewSpec returns a nil
// Spec and an error.
func NewSpec(raw *cdi.Spec, path string, priority int) (*Spec, error) {
	err := validateSpec(raw)
	if err != nil {
		return nil, err
	}

	spec := &Spec{
		Spec:     raw,
		path:     filepath.Clean(path),
		priority: priority,
	}

	spec.vendor, spec.class = ParseQualifier(spec.Kind)

	if spec.devices, err = spec.validate(); err != nil {
		return nil, errors.Wrap(err, "invalid CDI Spec")
	}

	return spec, nil
}

// GetVendor returns the vendor of this Spec.
func (s *Spec) GetVendor() string {
	return s.vendor
}

// GetClass returns the device class of this Spec.
func (s *Spec) GetClass() string {
	return s.class
}

// GetDevice returns the device for the given unqualified name.
func (s *Spec) GetDevice(name string) *Device {
	return s.devices[name]
}

// GetPath returns the filesystem path of this Spec.
func (s *Spec) GetPath() string {
	return s.path
}

// GetPriority returns the priority of this Spec.
func (s *Spec) GetPriority() int {
	return s.priority
}

// ApplyEdits applies the Spec's global-scope container edits to an OCI Spec.
func (s *Spec) ApplyEdits(ociSpec *oci.Spec) error {
	return s.edits().Apply(ociSpec)
}

// edits returns the applicable global container edits for this spec.
func (s *Spec) edits() *ContainerEdits {
	return &ContainerEdits{&s.ContainerEdits}
}

// Validate the Spec.
func (s *Spec) validate() (map[string]*Device, error) {
	if err := validateVersion(s.Version); err != nil {
		return nil, err
	}
	if err := ValidateVendorName(s.vendor); err != nil {
		return nil, err
	}
	if err := ValidateClassName(s.class); err != nil {
		return nil, err
	}
	if err := s.edits().Validate(); err != nil {
		return nil, err
	}

	devices := make(map[string]*Device)
	for _, d := range s.Devices {
		dev, err := newDevice(s, d)
		if err != nil {
			return nil, errors.Wrapf(err, "failed add device %q", d.Name)
		}
		if _, conflict := devices[d.Name]; conflict {
			return nil, errors.Errorf("invalid spec, multiple device %q", d.Name)
		}
		devices[d.Name] = dev
	}

	return devices, nil
}

// validateVersion checks whether the specified spec version is supported.
func validateVersion(version string) error {
	if _, ok := validSpecVersions[version]; !ok {
		return errors.Errorf("invalid version %q", version)
	}

	return nil
}

// Parse raw CDI Spec file data.
func parseSpec(data []byte) (*cdi.Spec, error) {
	var raw *cdi.Spec
	err := yaml.UnmarshalStrict(data, &raw)
	if err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal CDI Spec")
	}
	return raw, nil
}

// SetSpecValidator sets a CDI Spec validator function. This function
// is used for extra CDI Spec content validation whenever a Spec file
// loaded (using ReadSpec() or NewSpec()) or written (Spec.Write()).
func SetSpecValidator(fn func(*cdi.Spec) error) {
	specValidator = fn
}

// validateSpec validates the Spec using the extneral validator.
func validateSpec(raw *cdi.Spec) error {
	if specValidator == nil {
		return nil
	}
	err := specValidator(raw)
	if err != nil {
		return errors.Wrap(err, "Spec validation failed")
	}
	return nil
}
