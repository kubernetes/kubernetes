/*
Copyright 2021 The Kubernetes Authors.

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

// This package is meant to provide runtime/v1 type conversions into the
// internal intermediate types of this package.
package cri

import v1 "k8s.io/cri-api/pkg/apis/runtime/v1"

func FromV1VersionResponse(from *v1.VersionResponse) *VersionResponse {
	if from == nil {
		return nil
	}

	return &VersionResponse{
		Version:           from.Version,
		RuntimeName:       from.RuntimeName,
		RuntimeVersion:    from.RuntimeVersion,
		RuntimeApiVersion: from.RuntimeApiVersion,
	}
}

func FromV1PodSandboxStatus(from *v1.PodSandboxStatus) *PodSandboxStatus {
	if from == nil {
		return nil
	}

	to := &PodSandboxStatus{
		Id:             from.Id,
		State:          PodSandboxState(from.State),
		CreatedAt:      from.CreatedAt,
		Labels:         from.Labels,
		Annotations:    from.Annotations,
		RuntimeHandler: from.RuntimeHandler,
	}

	if from.Metadata != nil {
		to.Metadata = FromV1PodSandboxMetadata(from.Metadata)
	}
	if from.Network != nil {
		to.Network = &PodSandboxNetworkStatus{
			Ip: from.Network.Ip,
		}
		additionalIps := []*PodIP{}
		for _, x := range from.Network.AdditionalIps {
			additionalIps = append(additionalIps, &PodIP{Ip: x.Ip})
		}
		to.Network.AdditionalIps = additionalIps
	}
	if from.Linux != nil {
		to.Linux = &LinuxPodSandboxStatus{}
		if from.Linux.Namespaces != nil {
			to.Linux.Namespaces = &Namespace{}
			if from.Linux.Namespaces.Options != nil {
				to.Linux.Namespaces.Options = &NamespaceOption{
					Network:  NamespaceMode(from.Linux.Namespaces.Options.Network),
					Pid:      NamespaceMode(from.Linux.Namespaces.Options.Pid),
					Ipc:      NamespaceMode(from.Linux.Namespaces.Options.Ipc),
					TargetId: from.Linux.Namespaces.Options.TargetId,
				}
			}
		}
	}

	return to
}

func FromV1PodSandboxMetadata(from *v1.PodSandboxMetadata) *PodSandboxMetadata {
	if from == nil {
		return nil
	}
	return &PodSandboxMetadata{
		Name:      from.Name,
		Uid:       from.Uid,
		Namespace: from.Namespace,
		Attempt:   from.Attempt,
	}
}

func FromV1ContainerMetadata(from *v1.ContainerMetadata) *ContainerMetadata {
	if from == nil {
		return nil
	}
	return &ContainerMetadata{
		Name:    from.Name,
		Attempt: from.Attempt,
	}
}

func FromV1PodSandboxes(from []*v1.PodSandbox) (items []*PodSandbox) {
	for _, x := range from {
		if x == nil {
			continue
		}
		sandbox := &PodSandbox{
			Id:             x.Id,
			State:          PodSandboxState(x.State),
			CreatedAt:      x.CreatedAt,
			Labels:         x.Labels,
			Annotations:    x.Annotations,
			RuntimeHandler: x.RuntimeHandler,
		}
		if x.Metadata != nil {
			sandbox.Metadata = FromV1PodSandboxMetadata(x.Metadata)
		}

		items = append(items, sandbox)
	}
	return items
}

func FromV1Containers(from []*v1.Container) (items []*Container) {
	for _, x := range from {
		if x == nil {
			continue
		}
		container := &Container{
			Id:           x.Id,
			PodSandboxId: x.PodSandboxId,
			State:        ContainerState(x.State),
			ImageRef:     x.ImageRef,
			CreatedAt:    x.CreatedAt,
			Labels:       x.Labels,
			Annotations:  x.Annotations,
		}
		if x.Metadata != nil {
			container.Metadata = FromV1ContainerMetadata(x.Metadata)
		}
		if x.Image != nil {
			container.Image = &ImageSpec{
				Image:       x.Image.Image,
				Annotations: x.Image.Annotations,
			}
		}

		items = append(items, container)
	}
	return items
}

func FromV1ContainerStatus(from *v1.ContainerStatus) *ContainerStatus {
	if from == nil {
		return nil
	}

	to := &ContainerStatus{
		Id:          from.Id,
		State:       ContainerState(from.State),
		CreatedAt:   from.CreatedAt,
		StartedAt:   from.StartedAt,
		FinishedAt:  from.FinishedAt,
		ExitCode:    from.ExitCode,
		ImageRef:    from.ImageRef,
		Reason:      from.Reason,
		Message:     from.Message,
		Labels:      from.Labels,
		Annotations: from.Annotations,
		LogPath:     from.LogPath,
		Metadata:    &ContainerMetadata{},
		Image:       &ImageSpec{},
	}
	if from.Image != nil {
		to.Image = &ImageSpec{
			Image:       from.Image.Image,
			Annotations: from.Image.Annotations,
		}
	}
	if from.Metadata != nil {
		to.Metadata = FromV1ContainerMetadata(from.Metadata)
	}

	mounts := []*Mount{}
	for _, x := range from.Mounts {
		mounts = append(mounts, &Mount{
			ContainerPath:  x.ContainerPath,
			HostPath:       x.HostPath,
			Readonly:       x.Readonly,
			SelinuxRelabel: x.SelinuxRelabel,
			Propagation:    MountPropagation(x.Propagation),
		})
	}
	to.Mounts = mounts

	return to
}

func FromV1ExecResponse(from *v1.ExecResponse) *ExecResponse {
	if from == nil {
		return nil
	}

	return &ExecResponse{Url: from.Url}
}

func FromV1AttachResponse(from *v1.AttachResponse) *AttachResponse {
	if from == nil {
		return nil
	}

	return &AttachResponse{Url: from.Url}
}

func FromV1PortForwardResponse(from *v1.PortForwardResponse) *PortForwardResponse {
	if from == nil {
		return nil
	}

	return &PortForwardResponse{Url: from.Url}
}

func FromV1RuntimeStatus(from *v1.RuntimeStatus) *RuntimeStatus {
	if from == nil {
		return nil
	}

	conditions := []*RuntimeCondition{}
	for _, x := range from.Conditions {
		conditions = append(conditions, &RuntimeCondition{
			Type:    x.Type,
			Status:  x.Status,
			Reason:  x.Reason,
			Message: x.Message,
		})
	}

	return &RuntimeStatus{
		Conditions: conditions,
	}
}

func FromV1ContainerStats(from *v1.ContainerStats) *ContainerStats {
	if from == nil {
		return nil
	}

	to := &ContainerStats{}
	if from.Attributes != nil {
		to.Attributes = &ContainerAttributes{
			Id:          from.Attributes.Id,
			Labels:      from.Attributes.Labels,
			Annotations: from.Attributes.Annotations,
		}
		if from.Attributes.Metadata != nil {
			to.Attributes.Metadata = FromV1ContainerMetadata(from.Attributes.Metadata)
		}
	}
	if from.Cpu != nil {
		to.Cpu = &CpuUsage{
			Timestamp: from.Cpu.Timestamp,
		}
		if from.Cpu.UsageCoreNanoSeconds != nil {
			to.Cpu.UsageCoreNanoSeconds = &UInt64Value{
				Value: from.Cpu.UsageCoreNanoSeconds.Value,
			}
		}
	}
	if from.Memory != nil {
		to.Memory = &MemoryUsage{
			Timestamp: from.Memory.Timestamp,
		}
		if from.Memory.WorkingSetBytes != nil {
			to.Memory.WorkingSetBytes = &UInt64Value{
				Value: from.Memory.WorkingSetBytes.Value,
			}
		}
	}
	if from.WritableLayer != nil {
		to.WritableLayer = FromV1FilesystemUsage(from.WritableLayer)
	}

	return to
}

func FromV1FilesystemUsage(from *v1.FilesystemUsage) *FilesystemUsage {
	if from == nil {
		return nil
	}

	to := &FilesystemUsage{
		Timestamp: from.Timestamp,
	}
	if from.FsId != nil {
		to.FsId = &FilesystemIdentifier{
			Mountpoint: from.FsId.Mountpoint,
		}
	}
	if from.UsedBytes != nil {
		to.UsedBytes = &UInt64Value{
			Value: from.UsedBytes.Value,
		}
	}
	if from.InodesUsed != nil {
		to.InodesUsed = &UInt64Value{
			Value: from.InodesUsed.Value,
		}
	}

	return to
}

func FromV1FilesystemUsageList(from []*v1.FilesystemUsage) (items []*FilesystemUsage) {
	for _, x := range from {
		if x == nil {
			continue
		}
		items = append(items, FromV1FilesystemUsage(x))
	}

	return items
}

func FromV1ContainerStatsList(from []*v1.ContainerStats) (items []*ContainerStats) {
	for _, x := range from {
		if x == nil {
			continue
		}
		items = append(items, FromV1ContainerStats(x))
	}

	return items
}

func FromV1PodSandboxStats(from *v1.PodSandboxStats) *PodSandboxStats {
	if from == nil {
		return nil
	}

	to := &PodSandboxStats{}
	if from.Attributes != nil {
		to.Attributes = &PodSandboxAttributes{
			Id:          from.Attributes.Id,
			Labels:      from.Attributes.Labels,
			Annotations: from.Attributes.Annotations,
		}
		if from.Attributes.Metadata != nil {
			to.Attributes.Metadata = FromV1PodSandboxMetadata(from.Attributes.Metadata)
		}
	}
	if from.Linux != nil {
		to.Linux = &LinuxPodSandboxStats{}

		if from.Linux.Cpu != nil {
			to.Linux.Cpu = &CpuUsage{
				Timestamp: from.Linux.Cpu.Timestamp,
			}
			if from.Linux.Cpu.UsageCoreNanoSeconds != nil {
				to.Linux.Cpu.UsageCoreNanoSeconds = &UInt64Value{
					Value: from.Linux.Cpu.UsageCoreNanoSeconds.Value,
				}
			}
		}
		if from.Linux.Memory != nil {
			to.Linux.Memory = &MemoryUsage{
				Timestamp: from.Linux.Memory.Timestamp,
			}
			if from.Linux.Memory.WorkingSetBytes != nil {
				to.Linux.Memory.WorkingSetBytes = &UInt64Value{
					Value: from.Linux.Memory.WorkingSetBytes.Value,
				}
			}
		}
		if from.Linux.Network != nil {
			to.Linux.Network = &NetworkUsage{
				Timestamp: from.Linux.Network.Timestamp,
			}
			if from.Linux.Network.DefaultInterface != nil {
				to.Linux.Network.DefaultInterface = &NetworkInterfaceUsage{
					Name: from.Linux.Network.DefaultInterface.Name,
				}
				if from.Linux.Network.DefaultInterface.RxBytes != nil {
					to.Linux.Network.DefaultInterface.RxBytes = &UInt64Value{
						Value: from.Linux.Network.DefaultInterface.RxBytes.Value,
					}
				}
				if from.Linux.Network.DefaultInterface.RxErrors != nil {
					to.Linux.Network.DefaultInterface.RxErrors = &UInt64Value{
						Value: from.Linux.Network.DefaultInterface.RxErrors.Value,
					}
				}
				if from.Linux.Network.DefaultInterface.TxBytes != nil {
					to.Linux.Network.DefaultInterface.TxBytes = &UInt64Value{
						Value: from.Linux.Network.DefaultInterface.TxBytes.Value,
					}
				}
				if from.Linux.Network.DefaultInterface.TxErrors != nil {
					to.Linux.Network.DefaultInterface.TxErrors = &UInt64Value{
						Value: from.Linux.Network.DefaultInterface.TxErrors.Value,
					}
				}
			}
		}
		if from.Linux.Process != nil {
			to.Linux.Process = &ProcessUsage{
				Timestamp: from.Linux.Network.Timestamp,
			}
			if from.Linux.Process.ProcessCount != nil {
				to.Linux.Process.ProcessCount = &UInt64Value{
					Value: from.Linux.Process.ProcessCount.Value,
				}
			}
		}
	}

	return to
}

func FromV1PodSandboxStatsList(from []*v1.PodSandboxStats) (items []*PodSandboxStats) {
	for _, x := range from {
		if x == nil {
			continue
		}
		items = append(items, FromV1PodSandboxStats(x))
	}

	return items
}

func FromV1Image(from *v1.Image) *Image {
	if from == nil {
		return nil
	}

	to := &Image{
		Id:          from.Id,
		RepoTags:    from.RepoTags,
		RepoDigests: from.RepoDigests,
		Size_:       from.Size_,
		Username:    from.Username,
	}
	if from.Uid != nil {
		to.Uid = &Int64Value{
			Value: from.Uid.Value,
		}
	}
	if from.Spec != nil {
		to.Spec = &ImageSpec{
			Image:       from.Spec.Image,
			Annotations: from.Spec.Annotations,
		}
	}

	return to
}

func FromV1ImageList(from []*v1.Image) (items []*Image) {
	for _, x := range from {
		if x == nil {
			continue
		}
		items = append(items, FromV1Image(x))
	}

	return items
}

func V1PodSandboxConfig(from *PodSandboxConfig) *v1.PodSandboxConfig {
	if from == nil {
		return nil
	}
	to := &v1.PodSandboxConfig{
		Hostname:     from.Hostname,
		LogDirectory: from.LogDirectory,
		Labels:       from.Labels,
		Annotations:  from.Annotations,
		Linux: &v1.LinuxPodSandboxConfig{
			SecurityContext: NewV1LinuxSandboxSecurityContext(),
		},
	}

	if from.DnsConfig != nil {
		to.DnsConfig = &v1.DNSConfig{
			Servers:  from.DnsConfig.Servers,
			Searches: from.DnsConfig.Searches,
			Options:  from.DnsConfig.Options,
		}
	}
	if from.Metadata != nil {
		to.Metadata = &v1.PodSandboxMetadata{
			Name:      from.Metadata.Name,
			Uid:       from.Metadata.Uid,
			Namespace: from.Metadata.Namespace,
			Attempt:   from.Metadata.Attempt,
		}
	}
	portMappings := []*v1.PortMapping{}
	for _, x := range from.PortMappings {
		portMappings = append(portMappings, &v1.PortMapping{
			Protocol:      v1.Protocol(x.Protocol),
			ContainerPort: x.ContainerPort,
			HostPort:      x.HostPort,
			HostIp:        x.HostIp,
		})
	}
	to.PortMappings = portMappings
	if from.Linux != nil { // nolint: dupl
		to.Linux = &v1.LinuxPodSandboxConfig{
			CgroupParent:    from.Linux.CgroupParent,
			Sysctls:         from.Linux.Sysctls,
			SecurityContext: NewV1LinuxSandboxSecurityContext(),
		}
		if from.Linux.Overhead != nil {
			to.Linux.Overhead = &v1.LinuxContainerResources{
				CpuPeriod:          from.Linux.Overhead.CpuPeriod,
				CpuQuota:           from.Linux.Overhead.CpuQuota,
				CpuShares:          from.Linux.Overhead.CpuShares,
				MemoryLimitInBytes: from.Linux.Overhead.MemoryLimitInBytes,
				OomScoreAdj:        from.Linux.Overhead.OomScoreAdj,
				CpusetCpus:         from.Linux.Overhead.CpusetCpus,
				CpusetMems:         from.Linux.Overhead.CpusetMems,
			}
			hugepageLimits := []*v1.HugepageLimit{}
			for _, x := range from.Linux.Overhead.HugepageLimits {
				hugepageLimits = append(hugepageLimits, &v1.HugepageLimit{
					PageSize: x.PageSize,
					Limit:    x.Limit,
				})
			}
			to.Linux.Overhead.HugepageLimits = hugepageLimits
		}
		if from.Linux.Resources != nil {
			to.Linux.Resources = &v1.LinuxContainerResources{
				CpuPeriod:          from.Linux.Resources.CpuPeriod,
				CpuQuota:           from.Linux.Resources.CpuQuota,
				CpuShares:          from.Linux.Resources.CpuShares,
				MemoryLimitInBytes: from.Linux.Resources.MemoryLimitInBytes,
				OomScoreAdj:        from.Linux.Resources.OomScoreAdj,
				CpusetCpus:         from.Linux.Resources.CpusetCpus,
				CpusetMems:         from.Linux.Resources.CpusetMems,
			}
			hugepageLimits := []*v1.HugepageLimit{}
			for _, x := range from.Linux.Resources.HugepageLimits {
				hugepageLimits = append(hugepageLimits, &v1.HugepageLimit{
					PageSize: x.PageSize,
					Limit:    x.Limit,
				})
			}
			to.Linux.Resources.HugepageLimits = hugepageLimits
		}
		if from.Linux.SecurityContext != nil {
			to.Linux.SecurityContext = &v1.LinuxSandboxSecurityContext{
				SeccompProfilePath: from.Linux.SecurityContext.SeccompProfilePath,
				SupplementalGroups: from.Linux.SecurityContext.SupplementalGroups,
				ReadonlyRootfs:     from.Linux.SecurityContext.ReadonlyRootfs,
				Privileged:         from.Linux.SecurityContext.Privileged,
				NamespaceOptions:   &v1.NamespaceOption{},
				SelinuxOptions:     &v1.SELinuxOption{},
			}
			if from.Linux.SecurityContext.Seccomp != nil {
				to.Linux.SecurityContext.Seccomp = &v1.SecurityProfile{
					ProfileType:  v1.SecurityProfile_ProfileType(from.Linux.SecurityContext.Seccomp.ProfileType),
					LocalhostRef: from.Linux.SecurityContext.Seccomp.LocalhostRef,
				}
			}
			if from.Linux.SecurityContext.Apparmor != nil {
				to.Linux.SecurityContext.Apparmor = &v1.SecurityProfile{
					ProfileType:  v1.SecurityProfile_ProfileType(from.Linux.SecurityContext.Apparmor.ProfileType),
					LocalhostRef: from.Linux.SecurityContext.Apparmor.LocalhostRef,
				}
			}
			if from.Linux.SecurityContext.NamespaceOptions != nil {
				to.Linux.SecurityContext.NamespaceOptions = &v1.NamespaceOption{
					Network:  v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Network),
					Pid:      v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Pid),
					Ipc:      v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Ipc),
					TargetId: from.Linux.SecurityContext.NamespaceOptions.TargetId,
				}
			}
			if from.Linux.SecurityContext.SelinuxOptions != nil {
				to.Linux.SecurityContext.SelinuxOptions = &v1.SELinuxOption{
					User:  from.Linux.SecurityContext.SelinuxOptions.User,
					Role:  from.Linux.SecurityContext.SelinuxOptions.Role,
					Type:  from.Linux.SecurityContext.SelinuxOptions.Type,
					Level: from.Linux.SecurityContext.SelinuxOptions.Level,
				}
			}
			if from.Linux.SecurityContext.RunAsUser != nil {
				to.Linux.SecurityContext.RunAsUser = &v1.Int64Value{
					Value: from.Linux.SecurityContext.RunAsUser.Value,
				}
			}
			if from.Linux.SecurityContext.RunAsGroup != nil {
				to.Linux.SecurityContext.RunAsGroup = &v1.Int64Value{
					Value: from.Linux.SecurityContext.RunAsGroup.Value,
				}
			}
		}
	}

	return to
}

func NewV1LinuxSandboxSecurityContext() *v1.LinuxSandboxSecurityContext {
	return &v1.LinuxSandboxSecurityContext{
		NamespaceOptions: &v1.NamespaceOption{},
		SelinuxOptions:   &v1.SELinuxOption{},
		RunAsUser:        &v1.Int64Value{},
		RunAsGroup:       &v1.Int64Value{},
	}
}

func V1PodSandboxFilter(from *PodSandboxFilter) *v1.PodSandboxFilter {
	if from == nil {
		return nil
	}
	to := &v1.PodSandboxFilter{
		Id:            from.Id,
		LabelSelector: from.LabelSelector,
	}
	if from.State != nil {
		to.State = &v1.PodSandboxStateValue{
			State: v1.PodSandboxState(from.State.State),
		}
	}

	return to
}

func V1ContainerConfig(from *ContainerConfig) *v1.ContainerConfig {
	if from == nil {
		return nil
	}
	to := &v1.ContainerConfig{
		Command:     from.Command,
		Args:        from.Args,
		WorkingDir:  from.WorkingDir,
		Labels:      from.Labels,
		Annotations: from.Annotations,
		LogPath:     from.LogPath,
		Stdin:       from.Stdin,
		StdinOnce:   from.StdinOnce,
		Tty:         from.Tty,
		Linux:       NewV1LinuxContainerConfig(),
	}
	if from.Metadata != nil {
		to.Metadata = &v1.ContainerMetadata{
			Name:    from.Metadata.Name,
			Attempt: from.Metadata.Attempt,
		}
	}
	if from.Image != nil {
		to.Image = &v1.ImageSpec{
			Image:       from.Image.Image,
			Annotations: from.Image.Annotations,
		}
	}
	if from.Linux != nil {
		to.Linux = NewV1LinuxContainerConfig()
		if from.Linux.Resources != nil {
			to.Linux.Resources = &v1.LinuxContainerResources{
				CpuPeriod:          from.Linux.Resources.CpuPeriod,
				CpuQuota:           from.Linux.Resources.CpuQuota,
				CpuShares:          from.Linux.Resources.CpuShares,
				MemoryLimitInBytes: from.Linux.Resources.MemoryLimitInBytes,
				OomScoreAdj:        from.Linux.Resources.OomScoreAdj,
				CpusetCpus:         from.Linux.Resources.CpusetCpus,
				CpusetMems:         from.Linux.Resources.CpusetMems,
			}
			hugepageLimits := []*v1.HugepageLimit{}
			for _, x := range from.Linux.Resources.HugepageLimits {
				hugepageLimits = append(hugepageLimits, &v1.HugepageLimit{
					PageSize: x.PageSize,
					Limit:    x.Limit,
				})
			}
			to.Linux.Resources.HugepageLimits = hugepageLimits
		}
		if from.Linux.SecurityContext != nil {
			to.Linux.SecurityContext = &v1.LinuxContainerSecurityContext{
				RunAsUsername:      from.Linux.SecurityContext.RunAsUsername,
				ApparmorProfile:    from.Linux.SecurityContext.ApparmorProfile,
				SeccompProfilePath: from.Linux.SecurityContext.SeccompProfilePath,
				MaskedPaths:        from.Linux.SecurityContext.MaskedPaths,
				ReadonlyPaths:      from.Linux.SecurityContext.ReadonlyPaths,
				SupplementalGroups: from.Linux.SecurityContext.SupplementalGroups,
				Privileged:         from.Linux.SecurityContext.Privileged,
				ReadonlyRootfs:     from.Linux.SecurityContext.ReadonlyRootfs,
				NoNewPrivs:         from.Linux.SecurityContext.NoNewPrivs,
				Capabilities:       &v1.Capability{},
				NamespaceOptions:   &v1.NamespaceOption{},
				SelinuxOptions:     &v1.SELinuxOption{},
			}
			if from.Linux.SecurityContext.Capabilities != nil {
				to.Linux.SecurityContext.Capabilities = &v1.Capability{
					AddCapabilities:  from.Linux.SecurityContext.Capabilities.AddCapabilities,
					DropCapabilities: from.Linux.SecurityContext.Capabilities.DropCapabilities,
				}
			}
			if from.Linux.SecurityContext.Seccomp != nil {
				to.Linux.SecurityContext.Seccomp = &v1.SecurityProfile{
					ProfileType:  v1.SecurityProfile_ProfileType(from.Linux.SecurityContext.Seccomp.ProfileType),
					LocalhostRef: from.Linux.SecurityContext.Seccomp.LocalhostRef,
				}
			}
			if from.Linux.SecurityContext.Apparmor != nil {
				to.Linux.SecurityContext.Apparmor = &v1.SecurityProfile{
					ProfileType:  v1.SecurityProfile_ProfileType(from.Linux.SecurityContext.Apparmor.ProfileType),
					LocalhostRef: from.Linux.SecurityContext.Apparmor.LocalhostRef,
				}
			}
			if from.Linux.SecurityContext.NamespaceOptions != nil {
				to.Linux.SecurityContext.NamespaceOptions = &v1.NamespaceOption{
					Network:  v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Network),
					Pid:      v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Pid),
					Ipc:      v1.NamespaceMode(from.Linux.SecurityContext.NamespaceOptions.Ipc),
					TargetId: from.Linux.SecurityContext.NamespaceOptions.TargetId,
				}
			}
			if from.Linux.SecurityContext.SelinuxOptions != nil {
				to.Linux.SecurityContext.SelinuxOptions = &v1.SELinuxOption{
					User:  from.Linux.SecurityContext.SelinuxOptions.User,
					Role:  from.Linux.SecurityContext.SelinuxOptions.Role,
					Type:  from.Linux.SecurityContext.SelinuxOptions.Type,
					Level: from.Linux.SecurityContext.SelinuxOptions.Level,
				}
			}
			if from.Linux.SecurityContext.RunAsUser != nil {
				to.Linux.SecurityContext.RunAsUser = &v1.Int64Value{
					Value: from.Linux.SecurityContext.RunAsUser.Value,
				}
			}
			if from.Linux.SecurityContext.RunAsGroup != nil {
				to.Linux.SecurityContext.RunAsGroup = &v1.Int64Value{
					Value: from.Linux.SecurityContext.RunAsGroup.Value,
				}
			}
		}
	}
	envs := []*v1.KeyValue{}
	for _, x := range from.Envs {
		envs = append(envs, &v1.KeyValue{
			Key:   x.Key,
			Value: x.Value,
		})
	}
	to.Envs = envs

	mounts := []*v1.Mount{}
	for _, x := range from.Mounts {
		mounts = append(mounts, &v1.Mount{
			ContainerPath:  x.ContainerPath,
			HostPath:       x.HostPath,
			Readonly:       x.Readonly,
			SelinuxRelabel: x.SelinuxRelabel,
			Propagation:    v1.MountPropagation(x.Propagation),
		})
	}
	to.Mounts = mounts

	devices := []*v1.Device{}
	for _, x := range from.Devices {
		devices = append(devices, &v1.Device{
			ContainerPath: x.ContainerPath,
			HostPath:      x.HostPath,
			Permissions:   x.Permissions,
		})
	}
	to.Devices = devices

	return to
}

func NewV1LinuxContainerConfig() *v1.LinuxContainerConfig {
	return &v1.LinuxContainerConfig{
		Resources:       &v1.LinuxContainerResources{},
		SecurityContext: NewV1LinuxContainerSecurityContext(),
	}
}

func NewV1LinuxContainerSecurityContext() *v1.LinuxContainerSecurityContext {
	return &v1.LinuxContainerSecurityContext{
		Capabilities:     &v1.Capability{},
		NamespaceOptions: &v1.NamespaceOption{},
		SelinuxOptions:   &v1.SELinuxOption{},
		RunAsUser:        &v1.Int64Value{},
		RunAsGroup:       &v1.Int64Value{},
	}
}

func V1ContainerFilter(from *ContainerFilter) *v1.ContainerFilter {
	if from == nil {
		return nil
	}
	to := &v1.ContainerFilter{
		Id:            from.Id,
		LabelSelector: from.LabelSelector,
		PodSandboxId:  from.PodSandboxId,
	}
	if from.State != nil {
		to.State = &v1.ContainerStateValue{
			State: v1.ContainerState(from.State.State),
		}
	}

	return to
}

func V1ContainerResources(from *LinuxContainerResources) *v1.LinuxContainerResources {
	if from == nil {
		return nil
	}
	to := &v1.LinuxContainerResources{
		CpuPeriod:          from.CpuPeriod,
		CpuQuota:           from.CpuQuota,
		CpuShares:          from.CpuShares,
		MemoryLimitInBytes: from.MemoryLimitInBytes,
		OomScoreAdj:        from.OomScoreAdj,
		CpusetCpus:         from.CpusetCpus,
		CpusetMems:         from.CpusetMems,
	}
	hugePageLimits := []*v1.HugepageLimit{}
	for _, x := range from.HugepageLimits {
		hugePageLimits = append(hugePageLimits, &v1.HugepageLimit{
			PageSize: x.PageSize,
			Limit:    x.Limit,
		})
	}
	to.HugepageLimits = hugePageLimits

	return to
}

func V1ExecRequest(from *ExecRequest) *v1.ExecRequest {
	if from == nil {
		return nil
	}

	return &v1.ExecRequest{
		ContainerId: from.ContainerId,
		Cmd:         from.Cmd,
		Tty:         from.Tty,
		Stdin:       from.Stdin,
		Stdout:      from.Stdout,
		Stderr:      from.Stderr,
	}
}

func V1AttachRequest(from *AttachRequest) *v1.AttachRequest {
	if from == nil {
		return nil
	}

	return &v1.AttachRequest{
		ContainerId: from.ContainerId,
		Stdin:       from.Stdin,
		Tty:         from.Tty,
		Stdout:      from.Stdout,
		Stderr:      from.Stderr,
	}
}

func V1PortForwardRequest(from *PortForwardRequest) *v1.PortForwardRequest {
	if from == nil {
		return nil
	}

	return &v1.PortForwardRequest{
		PodSandboxId: from.PodSandboxId,
		Port:         from.Port,
	}
}

func V1RuntimeConfig(from *RuntimeConfig) *v1.RuntimeConfig {
	if from == nil {
		return nil
	}

	to := &v1.RuntimeConfig{}

	if from.NetworkConfig != nil {
		to.NetworkConfig = &v1.NetworkConfig{PodCidr: from.NetworkConfig.PodCidr}
	}

	return to
}

func V1ContainerStatsFilter(from *ContainerStatsFilter) *v1.ContainerStatsFilter {
	if from == nil {
		return nil
	}

	return &v1.ContainerStatsFilter{
		Id:            from.Id,
		LabelSelector: from.LabelSelector,
		PodSandboxId:  from.PodSandboxId,
	}
}

func V1PodSandboxStatsFilter(from *PodSandboxStatsFilter) *v1.PodSandboxStatsFilter {
	if from == nil {
		return nil
	}

	return &v1.PodSandboxStatsFilter{
		Id:            from.Id,
		LabelSelector: from.LabelSelector,
	}
}

func V1ImageFilter(from *ImageFilter) *v1.ImageFilter {
	if from == nil {
		return nil
	}

	to := &v1.ImageFilter{}

	if from.Image != nil {
		to.Image = V1ImageSpec(from.Image)
	}

	return to
}

func V1ImageSpec(from *ImageSpec) *v1.ImageSpec {
	if from == nil {
		return nil
	}

	return &v1.ImageSpec{
		Image:       from.Image,
		Annotations: from.Annotations,
	}
}

func V1AuthConfig(from *AuthConfig) *v1.AuthConfig {
	if from == nil {
		return nil
	}

	return &v1.AuthConfig{
		Username:      from.Username,
		Password:      from.Password,
		Auth:          from.Auth,
		ServerAddress: from.ServerAddress,
		IdentityToken: from.IdentityToken,
		RegistryToken: from.RegistryToken,
	}
}
