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
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/pkg/errors"

	"github.com/container-orchestrated-devices/container-device-interface/specs-go"
	oci "github.com/opencontainers/runtime-spec/specs-go"
	ocigen "github.com/opencontainers/runtime-tools/generate"

	runc "github.com/opencontainers/runc/libcontainer/devices"
)

const (
	// PrestartHook is the name of the OCI "prestart" hook.
	PrestartHook = "prestart"
	// CreateRuntimeHook is the name of the OCI "createRuntime" hook.
	CreateRuntimeHook = "createRuntime"
	// CreateContainerHook is the name of the OCI "createContainer" hook.
	CreateContainerHook = "createContainer"
	// StartContainerHook is the name of the OCI "startContainer" hook.
	StartContainerHook = "startContainer"
	// PoststartHook is the name of the OCI "poststart" hook.
	PoststartHook = "poststart"
	// PoststopHook is the name of the OCI "poststop" hook.
	PoststopHook = "poststop"
)

var (
	// Names of recognized hooks.
	validHookNames = map[string]struct{}{
		PrestartHook:        {},
		CreateRuntimeHook:   {},
		CreateContainerHook: {},
		StartContainerHook:  {},
		PoststartHook:       {},
		PoststopHook:        {},
	}
)

// ContainerEdits represent updates to be applied to an OCI Spec.
// These updates can be specific to a CDI device, or they can be
// specific to a CDI Spec. In the former case these edits should
// be applied to all OCI Specs where the corresponding CDI device
// is injected. In the latter case, these edits should be applied
// to all OCI Specs where at least one devices from the CDI Spec
// is injected.
type ContainerEdits struct {
	*specs.ContainerEdits
}

// Apply edits to the given OCI Spec. Updates the OCI Spec in place.
// Returns an error if the update fails.
func (e *ContainerEdits) Apply(spec *oci.Spec) error {
	if spec == nil {
		return errors.New("can't edit nil OCI Spec")
	}
	if e == nil || e.ContainerEdits == nil {
		return nil
	}

	specgen := ocigen.NewFromSpec(spec)
	if len(e.Env) > 0 {
		specgen.AddMultipleProcessEnv(e.Env)
	}

	for _, d := range e.DeviceNodes {
		dev := d.ToOCI()
		if err := fillMissingInfo(&dev); err != nil {
			return err
		}

		if dev.UID == nil && spec.Process != nil {
			if uid := spec.Process.User.UID; uid > 0 {
				dev.UID = &uid
			}
		}
		if dev.GID == nil && spec.Process != nil {
			if gid := spec.Process.User.GID; gid > 0 {
				dev.GID = &gid
			}
		}

		specgen.RemoveDevice(dev.Path)
		specgen.AddDevice(dev)

		if dev.Type == "b" || dev.Type == "c" {
			access := d.Permissions
			if access == "" {
				access = "rwm"
			}
			specgen.AddLinuxResourcesDevice(true, dev.Type, &dev.Major, &dev.Minor, access)
		}
	}

	if len(e.Mounts) > 0 {
		for _, m := range e.Mounts {
			specgen.RemoveMount(m.ContainerPath)
			specgen.AddMount(m.ToOCI())
		}
		sortMounts(&specgen)
	}

	for _, h := range e.Hooks {
		switch h.HookName {
		case PrestartHook:
			specgen.AddPreStartHook(h.ToOCI())
		case PoststartHook:
			specgen.AddPostStartHook(h.ToOCI())
		case PoststopHook:
			specgen.AddPostStopHook(h.ToOCI())
			// TODO: Maybe runtime-tools/generate should be updated with these...
		case CreateRuntimeHook:
			ensureOCIHooks(spec)
			spec.Hooks.CreateRuntime = append(spec.Hooks.CreateRuntime, h.ToOCI())
		case CreateContainerHook:
			ensureOCIHooks(spec)
			spec.Hooks.CreateContainer = append(spec.Hooks.CreateContainer, h.ToOCI())
		case StartContainerHook:
			ensureOCIHooks(spec)
			spec.Hooks.StartContainer = append(spec.Hooks.StartContainer, h.ToOCI())
		default:
			return errors.Errorf("unknown hook name %q", h.HookName)
		}
	}

	return nil
}

// Validate container edits.
func (e *ContainerEdits) Validate() error {
	if e == nil || e.ContainerEdits == nil {
		return nil
	}

	if err := ValidateEnv(e.Env); err != nil {
		return errors.Wrap(err, "invalid container edits")
	}
	for _, d := range e.DeviceNodes {
		if err := (&DeviceNode{d}).Validate(); err != nil {
			return err
		}
	}
	for _, h := range e.Hooks {
		if err := (&Hook{h}).Validate(); err != nil {
			return err
		}
	}
	for _, m := range e.Mounts {
		if err := (&Mount{m}).Validate(); err != nil {
			return err
		}
	}

	return nil
}

// Append other edits into this one. If called with a nil receiver,
// allocates and returns newly allocated edits.
func (e *ContainerEdits) Append(o *ContainerEdits) *ContainerEdits {
	if o == nil || o.ContainerEdits == nil {
		return e
	}
	if e == nil {
		e = &ContainerEdits{}
	}
	if e.ContainerEdits == nil {
		e.ContainerEdits = &specs.ContainerEdits{}
	}

	e.Env = append(e.Env, o.Env...)
	e.DeviceNodes = append(e.DeviceNodes, o.DeviceNodes...)
	e.Hooks = append(e.Hooks, o.Hooks...)
	e.Mounts = append(e.Mounts, o.Mounts...)

	return e
}

// isEmpty returns true if these edits are empty. This is valid in a
// global Spec context but invalid in a Device context.
func (e *ContainerEdits) isEmpty() bool {
	if e == nil {
		return false
	}
	return len(e.Env)+len(e.DeviceNodes)+len(e.Hooks)+len(e.Mounts) == 0
}

// ValidateEnv validates the given environment variables.
func ValidateEnv(env []string) error {
	for _, v := range env {
		if strings.IndexByte(v, byte('=')) <= 0 {
			return errors.Errorf("invalid environment variable %q", v)
		}
	}
	return nil
}

// DeviceNode is a CDI Spec DeviceNode wrapper, used for validating DeviceNodes.
type DeviceNode struct {
	*specs.DeviceNode
}

// Validate a CDI Spec DeviceNode.
func (d *DeviceNode) Validate() error {
	validTypes := map[string]struct{}{
		"":  {},
		"b": {},
		"c": {},
		"u": {},
		"p": {},
	}

	if d.Path == "" {
		return errors.New("invalid (empty) device path")
	}
	if _, ok := validTypes[d.Type]; !ok {
		return errors.Errorf("device %q: invalid type %q", d.Path, d.Type)
	}
	for _, bit := range d.Permissions {
		if bit != 'r' && bit != 'w' && bit != 'm' {
			return errors.Errorf("device %q: invalid persmissions %q",
				d.Path, d.Permissions)
		}
	}
	return nil
}

// Hook is a CDI Spec Hook wrapper, used for validating hooks.
type Hook struct {
	*specs.Hook
}

// Validate a hook.
func (h *Hook) Validate() error {
	if _, ok := validHookNames[h.HookName]; !ok {
		return errors.Errorf("invalid hook name %q", h.HookName)
	}
	if h.Path == "" {
		return errors.Errorf("invalid hook %q with empty path", h.HookName)
	}
	if err := ValidateEnv(h.Env); err != nil {
		return errors.Wrapf(err, "invalid hook %q", h.HookName)
	}
	return nil
}

// Mount is a CDI Mount wrapper, used for validating mounts.
type Mount struct {
	*specs.Mount
}

// Validate a mount.
func (m *Mount) Validate() error {
	if m.HostPath == "" {
		return errors.New("invalid mount, empty host path")
	}
	if m.ContainerPath == "" {
		return errors.New("invalid mount, empty container path")
	}
	return nil
}

// Ensure OCI Spec hooks are not nil so we can add hooks.
func ensureOCIHooks(spec *oci.Spec) {
	if spec.Hooks == nil {
		spec.Hooks = &oci.Hooks{}
	}
}

// fillMissingInfo fills in missing mandatory attributes from the host device.
func fillMissingInfo(dev *oci.LinuxDevice) error {
	if dev.Type != "" && (dev.Major != 0 || dev.Type == "p") {
		return nil
	}
	hostDev, err := runc.DeviceFromPath(dev.Path, "rwm")
	if err != nil {
		return errors.Wrapf(err, "failed to stat CDI host device %q", dev.Path)
	}

	if dev.Type == "" {
		dev.Type = string(hostDev.Type)
	} else {
		if dev.Type != string(hostDev.Type) {
			return errors.Errorf("CDI device %q, host type mismatch (%s, %s)",
				dev.Path, dev.Type, string(hostDev.Type))
		}
	}
	if dev.Major == 0 && dev.Type != "p" {
		dev.Major = hostDev.Major
		dev.Minor = hostDev.Minor
	}

	return nil
}

// sortMounts sorts the mounts in the given OCI Spec.
func sortMounts(specgen *ocigen.Generator) {
	mounts := specgen.Mounts()
	specgen.ClearMounts()
	sort.Sort(orderedMounts(mounts))
	specgen.Config.Mounts = mounts
}

// orderedMounts defines how to sort an OCI Spec Mount slice.
// This is the almost the same implementation sa used by CRI-O and Docker,
// with a minor tweak for stable sorting order (easier to test):
//   https://github.com/moby/moby/blob/17.05.x/daemon/volumes.go#L26
type orderedMounts []oci.Mount

// Len returns the number of mounts. Used in sorting.
func (m orderedMounts) Len() int {
	return len(m)
}

// Less returns true if the number of parts (a/b/c would be 3 parts) in the
// mount indexed by parameter 1 is less than that of the mount indexed by
// parameter 2. Used in sorting.
func (m orderedMounts) Less(i, j int) bool {
	ip, jp := m.parts(i), m.parts(j)
	if ip < jp {
		return true
	}
	if jp < ip {
		return false
	}
	return m[i].Destination < m[j].Destination
}

// Swap swaps two items in an array of mounts. Used in sorting
func (m orderedMounts) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

// parts returns the number of parts in the destination of a mount. Used in sorting.
func (m orderedMounts) parts(i int) int {
	return strings.Count(filepath.Clean(m[i].Destination), string(os.PathSeparator))
}
