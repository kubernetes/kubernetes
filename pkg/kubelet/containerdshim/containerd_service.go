/*
Copyright 2016 The Kubernetes Authors.

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

package containerdshim

import (
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/golang/glog"
	"google.golang.org/grpc"

	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"

	"github.com/docker/containerd/api/services/execution"
	"github.com/docker/containerd/api/services/rootfs"
	_ "github.com/docker/containerd/api/services/shim"
	_ "github.com/docker/containerd/api/types/container"
	_ "github.com/docker/containerd/api/types/mount"
	_ "github.com/opencontainers/image-spec/specs-go"
	_ "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	containerdVarLib  = "/var/lib/containerd"
	containerdVarRun  = "/var/run/containerd"
	containerdCRIRoot = "/tmp/containerd-cri"
	shimbindSocket    = "shim.sock"
)

type ContainerdService interface {
	internalapi.RuntimeService
	internalapi.ImageManagerService
	Start() error
	// For serving streaming calls.
	http.Handler
}

type containerdService struct {
	containerService execution.ContainerServiceClient
	rootfsService    rootfs.RootFSClient
}

func NewContainerdService(conn *grpc.ClientConn) ContainerdService {
	return &containerdService{
		containerService: execution.NewContainerServiceClient(conn),
		rootfsService:    rootfs.NewRootFSClient(conn),
	}
}

// The unix socket for containerdshhim <-> containerd communication.
const containerdBindSocket = "/run/containerd/containerd.sock" // mikebrow TODO get these from a config

// GetContainerdConnection returns a grpc client for containerd exection service.
func GetContainerdConnection() (*grpc.ClientConn, error) {
	// get the containerd client
	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithTimeout(100 * time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", containerdBindSocket, timeout)
		}),
	}
	return grpc.Dial(fmt.Sprintf("unix://%s", containerdBindSocket), dialOpts...)
}

// P4
func (cs *containerdService) Version(_ string) (*runtimeapi.VersionResponse, error) {
	return &runtimeapi.VersionResponse{
		Version:           "0.1.0",
		RuntimeName:       "containerd-poc",
		RuntimeVersion:    "1.0.0",
		RuntimeApiVersion: "1.0.0",
	}, nil
}

func (cs *containerdService) Start() error {
	glog.V(2).Infof("Start containerd service")
	return nil
}

// P4
func (cs *containerdService) UpdateRuntimeConfig(runtimeConfig *runtimeapi.RuntimeConfig) error {
	return nil
}

// P4
func (cs *containerdService) Status() (*runtimeapi.RuntimeStatus, error) {
	runtimeReady := &runtimeapi.RuntimeCondition{
		Type:   runtimeapi.RuntimeReady,
		Status: true,
	}
	networkReady := &runtimeapi.RuntimeCondition{
		Type:   runtimeapi.NetworkReady,
		Status: true,
	}
	return &runtimeapi.RuntimeStatus{Conditions: []*runtimeapi.RuntimeCondition{runtimeReady, networkReady}}, nil
}

// P3
func (cs *containerdService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	http.NotFound(w, r)
}
