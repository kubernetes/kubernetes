package convert

import (
	"strings"

	types "github.com/docker/docker/api/types/swarm"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
)

// TaskFromGRPC converts a grpc Task to a Task.
func TaskFromGRPC(t swarmapi.Task) (types.Task, error) {
	if t.Spec.GetAttachment() != nil {
		return types.Task{}, nil
	}
	containerStatus := t.Status.GetContainer()
	taskSpec, err := taskSpecFromGRPC(t.Spec)
	if err != nil {
		return types.Task{}, err
	}
	task := types.Task{
		ID:          t.ID,
		Annotations: annotationsFromGRPC(t.Annotations),
		ServiceID:   t.ServiceID,
		Slot:        int(t.Slot),
		NodeID:      t.NodeID,
		Spec:        taskSpec,
		Status: types.TaskStatus{
			State:   types.TaskState(strings.ToLower(t.Status.State.String())),
			Message: t.Status.Message,
			Err:     t.Status.Err,
		},
		DesiredState:     types.TaskState(strings.ToLower(t.DesiredState.String())),
		GenericResources: GenericResourcesFromGRPC(t.AssignedGenericResources),
	}

	// Meta
	task.Version.Index = t.Meta.Version.Index
	task.CreatedAt, _ = gogotypes.TimestampFromProto(t.Meta.CreatedAt)
	task.UpdatedAt, _ = gogotypes.TimestampFromProto(t.Meta.UpdatedAt)

	task.Status.Timestamp, _ = gogotypes.TimestampFromProto(t.Status.Timestamp)

	if containerStatus != nil {
		task.Status.ContainerStatus.ContainerID = containerStatus.ContainerID
		task.Status.ContainerStatus.PID = int(containerStatus.PID)
		task.Status.ContainerStatus.ExitCode = int(containerStatus.ExitCode)
	}

	// NetworksAttachments
	for _, na := range t.Networks {
		task.NetworksAttachments = append(task.NetworksAttachments, networkAttachmentFromGRPC(na))
	}

	if t.Status.PortStatus == nil {
		return task, nil
	}

	for _, p := range t.Status.PortStatus.Ports {
		task.Status.PortStatus.Ports = append(task.Status.PortStatus.Ports, types.PortConfig{
			Name:          p.Name,
			Protocol:      types.PortConfigProtocol(strings.ToLower(swarmapi.PortConfig_Protocol_name[int32(p.Protocol)])),
			PublishMode:   types.PortConfigPublishMode(strings.ToLower(swarmapi.PortConfig_PublishMode_name[int32(p.PublishMode)])),
			TargetPort:    p.TargetPort,
			PublishedPort: p.PublishedPort,
		})
	}

	return task, nil
}
