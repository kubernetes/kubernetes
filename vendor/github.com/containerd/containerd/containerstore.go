package containerd

import (
	"context"

	containersapi "github.com/containerd/containerd/api/services/containers/v1"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	ptypes "github.com/gogo/protobuf/types"
)

type remoteContainers struct {
	client containersapi.ContainersClient
}

var _ containers.Store = &remoteContainers{}

// NewRemoteContainerStore returns the container Store connected with the provided client
func NewRemoteContainerStore(client containersapi.ContainersClient) containers.Store {
	return &remoteContainers{
		client: client,
	}
}

func (r *remoteContainers) Get(ctx context.Context, id string) (containers.Container, error) {
	resp, err := r.client.Get(ctx, &containersapi.GetContainerRequest{
		ID: id,
	})
	if err != nil {
		return containers.Container{}, errdefs.FromGRPC(err)
	}

	return containerFromProto(&resp.Container), nil
}

func (r *remoteContainers) List(ctx context.Context, filters ...string) ([]containers.Container, error) {
	resp, err := r.client.List(ctx, &containersapi.ListContainersRequest{
		Filters: filters,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	return containersFromProto(resp.Containers), nil

}

func (r *remoteContainers) Create(ctx context.Context, container containers.Container) (containers.Container, error) {
	created, err := r.client.Create(ctx, &containersapi.CreateContainerRequest{
		Container: containerToProto(&container),
	})
	if err != nil {
		return containers.Container{}, errdefs.FromGRPC(err)
	}

	return containerFromProto(&created.Container), nil

}

func (r *remoteContainers) Update(ctx context.Context, container containers.Container, fieldpaths ...string) (containers.Container, error) {
	var updateMask *ptypes.FieldMask
	if len(fieldpaths) > 0 {
		updateMask = &ptypes.FieldMask{
			Paths: fieldpaths,
		}
	}

	updated, err := r.client.Update(ctx, &containersapi.UpdateContainerRequest{
		Container:  containerToProto(&container),
		UpdateMask: updateMask,
	})
	if err != nil {
		return containers.Container{}, errdefs.FromGRPC(err)
	}

	return containerFromProto(&updated.Container), nil

}

func (r *remoteContainers) Delete(ctx context.Context, id string) error {
	_, err := r.client.Delete(ctx, &containersapi.DeleteContainerRequest{
		ID: id,
	})

	return errdefs.FromGRPC(err)

}

func containerToProto(container *containers.Container) containersapi.Container {
	return containersapi.Container{
		ID:     container.ID,
		Labels: container.Labels,
		Image:  container.Image,
		Runtime: &containersapi.Container_Runtime{
			Name:    container.Runtime.Name,
			Options: container.Runtime.Options,
		},
		Spec:        container.Spec,
		Snapshotter: container.Snapshotter,
		SnapshotKey: container.SnapshotKey,
		Extensions:  container.Extensions,
	}
}

func containerFromProto(containerpb *containersapi.Container) containers.Container {
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
		CreatedAt:   containerpb.CreatedAt,
		UpdatedAt:   containerpb.UpdatedAt,
		Extensions:  containerpb.Extensions,
	}
}

func containersFromProto(containerspb []containersapi.Container) []containers.Container {
	var containers []containers.Container

	for _, container := range containerspb {
		containers = append(containers, containerFromProto(&container))
	}

	return containers
}
