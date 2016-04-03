package daemon

import (
	"fmt"

	"github.com/docker/docker/api/types"
)

func (daemon *Daemon) ContainerInspect(name string) (*types.ContainerJSON, error) {
	container, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}

	container.Lock()
	defer container.Unlock()

	base, err := daemon.getInspectData(container)
	if err != nil {
		return nil, err
	}

	mountPoints := make([]types.MountPoint, 0, len(container.MountPoints))
	for _, m := range container.MountPoints {
		mountPoints = append(mountPoints, types.MountPoint{
			Name:        m.Name,
			Source:      m.Path(),
			Destination: m.Destination,
			Driver:      m.Driver,
			Mode:        m.Relabel,
			RW:          m.RW,
		})
	}

	return &types.ContainerJSON{base, mountPoints, container.Config}, nil
}

func (daemon *Daemon) ContainerInspectPre120(name string) (*types.ContainerJSONPre120, error) {
	container, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}

	container.Lock()
	defer container.Unlock()

	base, err := daemon.getInspectData(container)
	if err != nil {
		return nil, err
	}

	volumes := make(map[string]string)
	volumesRW := make(map[string]bool)
	for _, m := range container.MountPoints {
		volumes[m.Destination] = m.Path()
		volumesRW[m.Destination] = m.RW
	}

	config := &types.ContainerConfig{
		container.Config,
		container.hostConfig.Memory,
		container.hostConfig.MemorySwap,
		container.hostConfig.CpuShares,
		container.hostConfig.CpusetCpus,
	}

	return &types.ContainerJSONPre120{base, volumes, volumesRW, config}, nil
}

func (daemon *Daemon) getInspectData(container *Container) (*types.ContainerJSONBase, error) {
	// make a copy to play with
	hostConfig := *container.hostConfig

	if children, err := daemon.Children(container.Name); err == nil {
		for linkAlias, child := range children {
			hostConfig.Links = append(hostConfig.Links, fmt.Sprintf("%s:%s", child.Name, linkAlias))
		}
	}
	// we need this trick to preserve empty log driver, so
	// container will use daemon defaults even if daemon change them
	if hostConfig.LogConfig.Type == "" {
		hostConfig.LogConfig = daemon.defaultLogConfig
	}

	containerState := &types.ContainerState{
		Running:    container.State.Running,
		Paused:     container.State.Paused,
		Restarting: container.State.Restarting,
		OOMKilled:  container.State.OOMKilled,
		Dead:       container.State.Dead,
		Pid:        container.State.Pid,
		ExitCode:   container.State.ExitCode,
		Error:      container.State.Error,
		StartedAt:  container.State.StartedAt,
		FinishedAt: container.State.FinishedAt,
	}

	contJSONBase := &types.ContainerJSONBase{
		Id:              container.ID,
		Created:         container.Created,
		Path:            container.Path,
		Args:            container.Args,
		State:           containerState,
		Image:           container.ImageID,
		NetworkSettings: container.NetworkSettings,
		ResolvConfPath:  container.ResolvConfPath,
		HostnamePath:    container.HostnamePath,
		HostsPath:       container.HostsPath,
		LogPath:         container.LogPath,
		Name:            container.Name,
		RestartCount:    container.RestartCount,
		Driver:          container.Driver,
		ExecDriver:      container.ExecDriver,
		MountLabel:      container.MountLabel,
		ProcessLabel:    container.ProcessLabel,
		AppArmorProfile: container.AppArmorProfile,
		ExecIDs:         container.GetExecIDs(),
		HostConfig:      &hostConfig,
	}

	contJSONBase.GraphDriver.Name = container.Driver
	graphDriverData, err := daemon.driver.GetMetadata(container.ID)
	if err != nil {
		return nil, err
	}
	contJSONBase.GraphDriver.Data = graphDriverData

	return contJSONBase, nil
}

func (daemon *Daemon) ContainerExecInspect(id string) (*execConfig, error) {
	eConfig, err := daemon.getExecConfig(id)
	if err != nil {
		return nil, err
	}
	return eConfig, nil
}
