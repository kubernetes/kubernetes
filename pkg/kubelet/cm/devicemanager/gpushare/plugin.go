/*
Copyright 2024 The Kubernetes Authors.

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

package gpushare

import (
	"context"
	"fmt"
	"net"
	"os"
	"path"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	resourceName = "gpushare.com/vgpu"
	serverSock   = pluginapi.DevicePluginPath + "gpushare.sock"
)

// GPUSharePlugin is a dummy device plugin that advertises fractional GPUs
// It simply multiplies physical GPUs by a factor (e.g., 100 slices per GPU)
type GPUSharePlugin struct {
	pluginapi.UnimplementedDevicePluginServer
	devs   []*pluginapi.Device
	socket string
	server *grpc.Server
	stop   chan struct{}
	wg     sync.WaitGroup
}

// NewGPUSharePlugin creates a new GPUShare device plugin
func NewGPUSharePlugin(numVirtualGPUs int) *GPUSharePlugin {
	var devs []*pluginapi.Device
	for i := 0; i < numVirtualGPUs; i++ {
		devs = append(devs, &pluginapi.Device{
			ID:     fmt.Sprintf("vgpu-%d", i),
			Health: pluginapi.Healthy,
		})
	}

	return &GPUSharePlugin{
		devs:   devs,
		socket: serverSock,
		stop:   make(chan struct{}),
	}
}

// Start starts the gRPC server and registers with the kubelet
func (p *GPUSharePlugin) Start() error {
	klog.Info("Starting GPUShare device plugin...")
	err := p.cleanup()
	if err != nil {
		return err
	}

	sock, err := net.Listen("unix", p.socket)
	if err != nil {
		return fmt.Errorf("starting device plugin server stopped, error listening on sock: %w", err)
	}

	p.server = grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterDevicePluginServer(p.server, p)

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		if err := p.server.Serve(sock); err != nil {
			klog.ErrorS(err, "GPUShare device plugin server crashed")
		}
	}()

	klog.Info("GPUShare device plugin started, waiting for ready...")
	// Wait for server to start by dialing it
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(ctx, p.socket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(c context.Context, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(c, "unix", addr)
		}),
	)
	if err != nil {
		return fmt.Errorf("could not wait for device plugin server to start: %w", err)
	}
	conn.Close()

	return p.register()
}

// Stop stops the gRPC server
func (p *GPUSharePlugin) Stop() error {
	klog.Info("Stopping GPUShare device plugin...")
	close(p.stop)
	if p.server != nil {
		p.server.Stop()
	}
	p.wg.Wait()
	return p.cleanup()
}

func (p *GPUSharePlugin) register() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(ctx, pluginapi.KubeletSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(c context.Context, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(c, "unix", addr)
		}),
	)
	if err != nil {
		return fmt.Errorf("device plugin unable to connect to Kubelet: %w", err)
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	reqt := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     path.Base(p.socket),
		ResourceName: resourceName,
	}

	_, err = client.Register(context.Background(), reqt)
	if err != nil {
		return fmt.Errorf("device plugin unable to register with Kubelet: %w", err)
	}
	klog.Info("GPUShare device plugin registered with Kubelet")
	return nil
}

func (p *GPUSharePlugin) cleanup() error {
	if err := os.Remove(p.socket); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("could not clean up socket %s: %w", p.socket, err)
	}
	return nil
}

// GetDevicePluginOptions returns options to be communicated with Device Manager
func (p *GPUSharePlugin) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{}, nil
}

// ListAndWatch returns a stream of List of Devices
func (p *GPUSharePlugin) ListAndWatch(e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	s.Send(&pluginapi.ListAndWatchResponse{Devices: p.devs})

	<-p.stop
	return nil
}

// Allocate is called during container creation so that the Device Plugin can run device specific operations
func (p *GPUSharePlugin) Allocate(ctx context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	responses := pluginapi.AllocateResponse{}
	for range reqs.ContainerRequests {
		response := pluginapi.ContainerAllocateResponse{
			Envs: map[string]string{
				"NVIDIA_VISIBLE_DEVICES": "all", // Pass through logic for MPS or time-slicing
				"GPU_SHARE_ALLOCATED":    "true",
			},
		}
		responses.ContainerResponses = append(responses.ContainerResponses, &response)
	}
	return &responses, nil
}

// PreStartContainer is called, if indicated by Device Plugin during registration phase, before each container start.
func (p *GPUSharePlugin) PreStartContainer(context.Context, *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

// GetPreferredAllocation returns a preferred set of devices to allocate from a list of available ones.
func (p *GPUSharePlugin) GetPreferredAllocation(context.Context, *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}
