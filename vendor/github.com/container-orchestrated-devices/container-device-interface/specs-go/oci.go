package specs

import (
	"errors"
	"fmt"

	spec "github.com/opencontainers/runtime-spec/specs-go"
)

// ApplyOCIEditsForDevice applies devices OCI edits, in other words
// it finds the device in the CDI spec and applies the OCI patches that device
// requires to the OCI specification.
func ApplyOCIEditsForDevice(config *spec.Spec, cdi *Spec, dev string) error {
	for _, d := range cdi.Devices {
		if d.Name != dev {
			continue
		}

		return ApplyEditsToOCISpec(config, &d.ContainerEdits)
	}

	return fmt.Errorf("CDI: device %q not found for spec %q", dev, cdi.Kind)
}

// ApplyOCIEdits applies the OCI edits the CDI spec declares globablly
func ApplyOCIEdits(config *spec.Spec, cdi *Spec) error {
	return ApplyEditsToOCISpec(config, &cdi.ContainerEdits)
}

// ApplyEditsToOCISpec applies the specified edits to the OCI spec.
func ApplyEditsToOCISpec(config *spec.Spec, edits *ContainerEdits) error {
	if config == nil {
		return errors.New("spec is nil")
	}
	if edits == nil {
		return nil
	}

	if len(edits.Env) > 0 {
		if config.Process == nil {
			config.Process = &spec.Process{}
		}
		config.Process.Env = append(config.Process.Env, edits.Env...)
	}

	for _, d := range edits.DeviceNodes {
		if config.Linux == nil {
			config.Linux = &spec.Linux{}
		}
		config.Linux.Devices = append(config.Linux.Devices, d.ToOCI())
	}

	for _, m := range edits.Mounts {
		config.Mounts = append(config.Mounts, m.ToOCI())
	}

	for _, h := range edits.Hooks {
		if config.Hooks == nil {
			config.Hooks = &spec.Hooks{}
		}
		switch h.HookName {
		case "prestart":
			config.Hooks.Prestart = append(config.Hooks.Prestart, h.ToOCI())
		case "createRuntime":
			config.Hooks.CreateRuntime = append(config.Hooks.CreateRuntime, h.ToOCI())
		case "createContainer":
			config.Hooks.CreateContainer = append(config.Hooks.CreateContainer, h.ToOCI())
		case "startContainer":
			config.Hooks.StartContainer = append(config.Hooks.StartContainer, h.ToOCI())
		case "poststart":
			config.Hooks.Poststart = append(config.Hooks.Poststart, h.ToOCI())
		case "poststop":
			config.Hooks.Poststop = append(config.Hooks.Poststop, h.ToOCI())
		default:
			fmt.Printf("CDI: Unknown hook %q\n", h.HookName)
		}
	}

	return nil
}

// ToOCI returns the opencontainers runtime Spec Hook for this Hook.
func (h *Hook) ToOCI() spec.Hook {
	return spec.Hook{
		Path:    h.Path,
		Args:    h.Args,
		Env:     h.Env,
		Timeout: h.Timeout,
	}
}

// ToOCI returns the opencontainers runtime Spec Mount for this Mount.
func (m *Mount) ToOCI() spec.Mount {
	return spec.Mount{
		Source:      m.HostPath,
		Destination: m.ContainerPath,
		Options:     m.Options,
		Type:        m.Type,
	}
}

// ToOCI returns the opencontainers runtime Spec LinuxDevice for this DeviceNode.
func (d *DeviceNode) ToOCI() spec.LinuxDevice {
	return spec.LinuxDevice{
		Path:     d.Path,
		Type:     d.Type,
		Major:    d.Major,
		Minor:    d.Minor,
		FileMode: d.FileMode,
		UID:      d.UID,
		GID:      d.GID,
	}
}
