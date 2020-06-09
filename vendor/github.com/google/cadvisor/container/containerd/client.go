// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package containerd

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	containersapi "github.com/containerd/containerd/api/services/containers/v1"
	tasksapi "github.com/containerd/containerd/api/services/tasks/v1"
	versionapi "github.com/containerd/containerd/api/services/version/v1"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/pkg/dialer"
	ptypes "github.com/gogo/protobuf/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
)

type client struct {
	containerService containersapi.ContainersClient
	taskService      tasksapi.TasksClient
	versionService   versionapi.VersionClient
}

type ContainerdClient interface {
	LoadContainer(ctx context.Context, id string) (*containers.Container, error)
	TaskPid(ctx context.Context, id string) (uint32, error)
	Version(ctx context.Context) (string, error)
}

var once sync.Once
var ctrdClient ContainerdClient = nil

const (
	maxBackoffDelay   = 3 * time.Second
	baseBackoffDelay  = 100 * time.Millisecond
	connectionTimeout = 2 * time.Second
)

// Client creates a containerd client
func Client(address, namespace string) (ContainerdClient, error) {
	var retErr error
	once.Do(func() {
		tryConn, err := net.DialTimeout("unix", address, connectionTimeout)
		if err != nil {
			retErr = fmt.Errorf("containerd: cannot unix dial containerd api service: %v", err)
			return
		}
		tryConn.Close()

		connParams := grpc.ConnectParams{
			Backoff: backoff.Config{
				BaseDelay: baseBackoffDelay,
				MaxDelay:  maxBackoffDelay,
			},
		}
		gopts := []grpc.DialOption{
			grpc.WithInsecure(),
			grpc.WithContextDialer(dialer.ContextDialer),
			grpc.WithBlock(),
			grpc.WithConnectParams(connParams),
		}
		unary, stream := newNSInterceptors(namespace)
		gopts = append(gopts,
			grpc.WithUnaryInterceptor(unary),
			grpc.WithStreamInterceptor(stream),
		)

		ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
		defer cancel()
		conn, err := grpc.DialContext(ctx, dialer.DialAddress(address), gopts...)
		if err != nil {
			retErr = err
			return
		}
		ctrdClient = &client{
			containerService: containersapi.NewContainersClient(conn),
			taskService:      tasksapi.NewTasksClient(conn),
			versionService:   versionapi.NewVersionClient(conn),
		}
	})
	return ctrdClient, retErr
}

func (c *client) LoadContainer(ctx context.Context, id string) (*containers.Container, error) {
	r, err := c.containerService.Get(ctx, &containersapi.GetContainerRequest{
		ID: id,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return containerFromProto(r.Container), nil
}

func (c *client) TaskPid(ctx context.Context, id string) (uint32, error) {
	response, err := c.taskService.Get(ctx, &tasksapi.GetRequest{
		ContainerID: id,
	})
	if err != nil {
		return 0, errdefs.FromGRPC(err)
	}
	return response.Process.Pid, nil
}

func (c *client) Version(ctx context.Context) (string, error) {
	response, err := c.versionService.Version(ctx, &ptypes.Empty{})
	if err != nil {
		return "", errdefs.FromGRPC(err)
	}
	return response.Version, nil
}

func containerFromProto(containerpb containersapi.Container) *containers.Container {
	var runtime containers.RuntimeInfo
	if containerpb.Runtime != nil {
		runtime = containers.RuntimeInfo{
			Name:    containerpb.Runtime.Name,
			Options: containerpb.Runtime.Options,
		}
	}
	return &containers.Container{
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
