package containers

import (
	api "github.com/containerd/containerd/api/services/containers/v1"
	"github.com/containerd/containerd/containers"
)

func containersToProto(containers []containers.Container) []api.Container {
	var containerspb []api.Container

	for _, image := range containers {
		containerspb = append(containerspb, containerToProto(&image))
	}

	return containerspb
}

func containerToProto(container *containers.Container) api.Container {
	return api.Container{
		ID:     container.ID,
		Labels: container.Labels,
		Image:  container.Image,
		Runtime: &api.Container_Runtime{
			Name:    container.Runtime.Name,
			Options: container.Runtime.Options,
		},
		Spec:        container.Spec,
		Snapshotter: container.Snapshotter,
		SnapshotKey: container.SnapshotKey,
		CreatedAt:   container.CreatedAt,
		UpdatedAt:   container.UpdatedAt,
		Extensions:  container.Extensions,
	}
}

func containerFromProto(containerpb *api.Container) containers.Container {
	var runtime containers.RuntimeInfo
	if containerpb.Runtime != nil {
		runtime = containers.RuntimeInfo{
			Name:    containerpb.Runtime.Name,
			Options: containerpb.Runtime.Options,
		}
	}
	return containers.Container{
		ID:          containerpb.ID,
		Labels:      containerpb.Labels,
		Image:       containerpb.Image,
		Runtime:     runtime,
		Spec:        containerpb.Spec,
		Snapshotter: containerpb.Snapshotter,
		SnapshotKey: containerpb.SnapshotKey,
		Extensions:  containerpb.Extensions,
	}
}
