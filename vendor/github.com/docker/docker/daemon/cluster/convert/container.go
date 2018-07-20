package convert

import (
	"errors"
	"fmt"
	"strings"

	container "github.com/docker/docker/api/types/container"
	mounttypes "github.com/docker/docker/api/types/mount"
	types "github.com/docker/docker/api/types/swarm"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
	"github.com/sirupsen/logrus"
)

func containerSpecFromGRPC(c *swarmapi.ContainerSpec) *types.ContainerSpec {
	if c == nil {
		return nil
	}
	containerSpec := &types.ContainerSpec{
		Image:      c.Image,
		Labels:     c.Labels,
		Command:    c.Command,
		Args:       c.Args,
		Hostname:   c.Hostname,
		Env:        c.Env,
		Dir:        c.Dir,
		User:       c.User,
		Groups:     c.Groups,
		StopSignal: c.StopSignal,
		TTY:        c.TTY,
		OpenStdin:  c.OpenStdin,
		ReadOnly:   c.ReadOnly,
		Hosts:      c.Hosts,
		Secrets:    secretReferencesFromGRPC(c.Secrets),
		Configs:    configReferencesFromGRPC(c.Configs),
		Isolation:  IsolationFromGRPC(c.Isolation),
	}

	if c.DNSConfig != nil {
		containerSpec.DNSConfig = &types.DNSConfig{
			Nameservers: c.DNSConfig.Nameservers,
			Search:      c.DNSConfig.Search,
			Options:     c.DNSConfig.Options,
		}
	}

	// Privileges
	if c.Privileges != nil {
		containerSpec.Privileges = &types.Privileges{}

		if c.Privileges.CredentialSpec != nil {
			containerSpec.Privileges.CredentialSpec = &types.CredentialSpec{}
			switch c.Privileges.CredentialSpec.Source.(type) {
			case *swarmapi.Privileges_CredentialSpec_File:
				containerSpec.Privileges.CredentialSpec.File = c.Privileges.CredentialSpec.GetFile()
			case *swarmapi.Privileges_CredentialSpec_Registry:
				containerSpec.Privileges.CredentialSpec.Registry = c.Privileges.CredentialSpec.GetRegistry()
			}
		}

		if c.Privileges.SELinuxContext != nil {
			containerSpec.Privileges.SELinuxContext = &types.SELinuxContext{
				Disable: c.Privileges.SELinuxContext.Disable,
				User:    c.Privileges.SELinuxContext.User,
				Type:    c.Privileges.SELinuxContext.Type,
				Role:    c.Privileges.SELinuxContext.Role,
				Level:   c.Privileges.SELinuxContext.Level,
			}
		}
	}

	// Mounts
	for _, m := range c.Mounts {
		mount := mounttypes.Mount{
			Target:   m.Target,
			Source:   m.Source,
			Type:     mounttypes.Type(strings.ToLower(swarmapi.Mount_MountType_name[int32(m.Type)])),
			ReadOnly: m.ReadOnly,
		}

		if m.BindOptions != nil {
			mount.BindOptions = &mounttypes.BindOptions{
				Propagation: mounttypes.Propagation(strings.ToLower(swarmapi.Mount_BindOptions_MountPropagation_name[int32(m.BindOptions.Propagation)])),
			}
		}

		if m.VolumeOptions != nil {
			mount.VolumeOptions = &mounttypes.VolumeOptions{
				NoCopy: m.VolumeOptions.NoCopy,
				Labels: m.VolumeOptions.Labels,
			}
			if m.VolumeOptions.DriverConfig != nil {
				mount.VolumeOptions.DriverConfig = &mounttypes.Driver{
					Name:    m.VolumeOptions.DriverConfig.Name,
					Options: m.VolumeOptions.DriverConfig.Options,
				}
			}
		}

		if m.TmpfsOptions != nil {
			mount.TmpfsOptions = &mounttypes.TmpfsOptions{
				SizeBytes: m.TmpfsOptions.SizeBytes,
				Mode:      m.TmpfsOptions.Mode,
			}
		}
		containerSpec.Mounts = append(containerSpec.Mounts, mount)
	}

	if c.StopGracePeriod != nil {
		grace, _ := gogotypes.DurationFromProto(c.StopGracePeriod)
		containerSpec.StopGracePeriod = &grace
	}

	if c.Healthcheck != nil {
		containerSpec.Healthcheck = healthConfigFromGRPC(c.Healthcheck)
	}

	return containerSpec
}

func secretReferencesToGRPC(sr []*types.SecretReference) []*swarmapi.SecretReference {
	refs := make([]*swarmapi.SecretReference, 0, len(sr))
	for _, s := range sr {
		ref := &swarmapi.SecretReference{
			SecretID:   s.SecretID,
			SecretName: s.SecretName,
		}
		if s.File != nil {
			ref.Target = &swarmapi.SecretReference_File{
				File: &swarmapi.FileTarget{
					Name: s.File.Name,
					UID:  s.File.UID,
					GID:  s.File.GID,
					Mode: s.File.Mode,
				},
			}
		}

		refs = append(refs, ref)
	}

	return refs
}

func secretReferencesFromGRPC(sr []*swarmapi.SecretReference) []*types.SecretReference {
	refs := make([]*types.SecretReference, 0, len(sr))
	for _, s := range sr {
		target := s.GetFile()
		if target == nil {
			// not a file target
			logrus.Warnf("secret target not a file: secret=%s", s.SecretID)
			continue
		}
		refs = append(refs, &types.SecretReference{
			File: &types.SecretReferenceFileTarget{
				Name: target.Name,
				UID:  target.UID,
				GID:  target.GID,
				Mode: target.Mode,
			},
			SecretID:   s.SecretID,
			SecretName: s.SecretName,
		})
	}

	return refs
}

func configReferencesToGRPC(sr []*types.ConfigReference) []*swarmapi.ConfigReference {
	refs := make([]*swarmapi.ConfigReference, 0, len(sr))
	for _, s := range sr {
		ref := &swarmapi.ConfigReference{
			ConfigID:   s.ConfigID,
			ConfigName: s.ConfigName,
		}
		if s.File != nil {
			ref.Target = &swarmapi.ConfigReference_File{
				File: &swarmapi.FileTarget{
					Name: s.File.Name,
					UID:  s.File.UID,
					GID:  s.File.GID,
					Mode: s.File.Mode,
				},
			}
		}

		refs = append(refs, ref)
	}

	return refs
}

func configReferencesFromGRPC(sr []*swarmapi.ConfigReference) []*types.ConfigReference {
	refs := make([]*types.ConfigReference, 0, len(sr))
	for _, s := range sr {
		target := s.GetFile()
		if target == nil {
			// not a file target
			logrus.Warnf("config target not a file: config=%s", s.ConfigID)
			continue
		}
		refs = append(refs, &types.ConfigReference{
			File: &types.ConfigReferenceFileTarget{
				Name: target.Name,
				UID:  target.UID,
				GID:  target.GID,
				Mode: target.Mode,
			},
			ConfigID:   s.ConfigID,
			ConfigName: s.ConfigName,
		})
	}

	return refs
}

func containerToGRPC(c *types.ContainerSpec) (*swarmapi.ContainerSpec, error) {
	containerSpec := &swarmapi.ContainerSpec{
		Image:      c.Image,
		Labels:     c.Labels,
		Command:    c.Command,
		Args:       c.Args,
		Hostname:   c.Hostname,
		Env:        c.Env,
		Dir:        c.Dir,
		User:       c.User,
		Groups:     c.Groups,
		StopSignal: c.StopSignal,
		TTY:        c.TTY,
		OpenStdin:  c.OpenStdin,
		ReadOnly:   c.ReadOnly,
		Hosts:      c.Hosts,
		Secrets:    secretReferencesToGRPC(c.Secrets),
		Configs:    configReferencesToGRPC(c.Configs),
		Isolation:  isolationToGRPC(c.Isolation),
	}

	if c.DNSConfig != nil {
		containerSpec.DNSConfig = &swarmapi.ContainerSpec_DNSConfig{
			Nameservers: c.DNSConfig.Nameservers,
			Search:      c.DNSConfig.Search,
			Options:     c.DNSConfig.Options,
		}
	}

	if c.StopGracePeriod != nil {
		containerSpec.StopGracePeriod = gogotypes.DurationProto(*c.StopGracePeriod)
	}

	// Privileges
	if c.Privileges != nil {
		containerSpec.Privileges = &swarmapi.Privileges{}

		if c.Privileges.CredentialSpec != nil {
			containerSpec.Privileges.CredentialSpec = &swarmapi.Privileges_CredentialSpec{}

			if c.Privileges.CredentialSpec.File != "" && c.Privileges.CredentialSpec.Registry != "" {
				return nil, errors.New("cannot specify both \"file\" and \"registry\" credential specs")
			}
			if c.Privileges.CredentialSpec.File != "" {
				containerSpec.Privileges.CredentialSpec.Source = &swarmapi.Privileges_CredentialSpec_File{
					File: c.Privileges.CredentialSpec.File,
				}
			} else if c.Privileges.CredentialSpec.Registry != "" {
				containerSpec.Privileges.CredentialSpec.Source = &swarmapi.Privileges_CredentialSpec_Registry{
					Registry: c.Privileges.CredentialSpec.Registry,
				}
			} else {
				return nil, errors.New("must either provide \"file\" or \"registry\" for credential spec")
			}
		}

		if c.Privileges.SELinuxContext != nil {
			containerSpec.Privileges.SELinuxContext = &swarmapi.Privileges_SELinuxContext{
				Disable: c.Privileges.SELinuxContext.Disable,
				User:    c.Privileges.SELinuxContext.User,
				Type:    c.Privileges.SELinuxContext.Type,
				Role:    c.Privileges.SELinuxContext.Role,
				Level:   c.Privileges.SELinuxContext.Level,
			}
		}
	}

	// Mounts
	for _, m := range c.Mounts {
		mount := swarmapi.Mount{
			Target:   m.Target,
			Source:   m.Source,
			ReadOnly: m.ReadOnly,
		}

		if mountType, ok := swarmapi.Mount_MountType_value[strings.ToUpper(string(m.Type))]; ok {
			mount.Type = swarmapi.Mount_MountType(mountType)
		} else if string(m.Type) != "" {
			return nil, fmt.Errorf("invalid MountType: %q", m.Type)
		}

		if m.BindOptions != nil {
			if mountPropagation, ok := swarmapi.Mount_BindOptions_MountPropagation_value[strings.ToUpper(string(m.BindOptions.Propagation))]; ok {
				mount.BindOptions = &swarmapi.Mount_BindOptions{Propagation: swarmapi.Mount_BindOptions_MountPropagation(mountPropagation)}
			} else if string(m.BindOptions.Propagation) != "" {
				return nil, fmt.Errorf("invalid MountPropagation: %q", m.BindOptions.Propagation)
			}
		}

		if m.VolumeOptions != nil {
			mount.VolumeOptions = &swarmapi.Mount_VolumeOptions{
				NoCopy: m.VolumeOptions.NoCopy,
				Labels: m.VolumeOptions.Labels,
			}
			if m.VolumeOptions.DriverConfig != nil {
				mount.VolumeOptions.DriverConfig = &swarmapi.Driver{
					Name:    m.VolumeOptions.DriverConfig.Name,
					Options: m.VolumeOptions.DriverConfig.Options,
				}
			}
		}

		if m.TmpfsOptions != nil {
			mount.TmpfsOptions = &swarmapi.Mount_TmpfsOptions{
				SizeBytes: m.TmpfsOptions.SizeBytes,
				Mode:      m.TmpfsOptions.Mode,
			}
		}

		containerSpec.Mounts = append(containerSpec.Mounts, mount)
	}

	if c.Healthcheck != nil {
		containerSpec.Healthcheck = healthConfigToGRPC(c.Healthcheck)
	}

	return containerSpec, nil
}

func healthConfigFromGRPC(h *swarmapi.HealthConfig) *container.HealthConfig {
	interval, _ := gogotypes.DurationFromProto(h.Interval)
	timeout, _ := gogotypes.DurationFromProto(h.Timeout)
	startPeriod, _ := gogotypes.DurationFromProto(h.StartPeriod)
	return &container.HealthConfig{
		Test:        h.Test,
		Interval:    interval,
		Timeout:     timeout,
		Retries:     int(h.Retries),
		StartPeriod: startPeriod,
	}
}

func healthConfigToGRPC(h *container.HealthConfig) *swarmapi.HealthConfig {
	return &swarmapi.HealthConfig{
		Test:        h.Test,
		Interval:    gogotypes.DurationProto(h.Interval),
		Timeout:     gogotypes.DurationProto(h.Timeout),
		Retries:     int32(h.Retries),
		StartPeriod: gogotypes.DurationProto(h.StartPeriod),
	}
}

// IsolationFromGRPC converts a swarm api container isolation to a moby isolation representation
func IsolationFromGRPC(i swarmapi.ContainerSpec_Isolation) container.Isolation {
	switch i {
	case swarmapi.ContainerIsolationHyperV:
		return container.IsolationHyperV
	case swarmapi.ContainerIsolationProcess:
		return container.IsolationProcess
	case swarmapi.ContainerIsolationDefault:
		return container.IsolationDefault
	}
	return container.IsolationEmpty
}

func isolationToGRPC(i container.Isolation) swarmapi.ContainerSpec_Isolation {
	if i.IsHyperV() {
		return swarmapi.ContainerIsolationHyperV
	}
	if i.IsProcess() {
		return swarmapi.ContainerIsolationProcess
	}
	return swarmapi.ContainerIsolationDefault
}
