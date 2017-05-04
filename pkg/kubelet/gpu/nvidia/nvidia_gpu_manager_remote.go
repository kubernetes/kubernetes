/*
Copyright 2017 The Kubernetes Authors.

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

package nvidia

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/gpu"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/gpu/api/v1alpha1/runtime"
)

const (
	defaultTimeout = time.Second * 5
)

// nvidiaGPUManager manages nvidia gpu devices.
type remoteGPUManager struct {
	sync.Mutex
	client runtimeapi.GPUAllocatorServiceClient
}

func init() {
	gpu.RegisterGPUManagerInitializer("remote", NewRemoteGPUManager)
}

// NewRemoteGPUManager returns a GPUManager that manages local GPUs.
// TODO: Migrate to use pod level cgroups and make it generic to all runtimes.
func NewRemoteGPUManager(_ gpu.ActivePodsLister,
	_ dockertools.DockerInterface,
	endpoint string) (gpu.GPUManager, error) {

	if endpoint == "" {
		return nil, fmt.Errorf("empty endpoint address")
	}

	rgm := &remoteGPUManager{}
	conn, err := grpc.Dial(endpoint, grpc.WithInsecure(), grpc.WithDialer(dial), grpc.WithTimeout(defaultTimeout))

	if err != nil {
		return nil, err
	}

	rgm.client = runtimeapi.NewGPUAllocatorServiceClient(conn)

	return rgm, nil
}

// Initialize the GPU devices, so far only needed to discover the GPU paths.
func (rgm *remoteGPUManager) Start() error {
	return nil
}

// Get how many GPU cards we have.
func (rgm *remoteGPUManager) Capacity() v1.ResourceList {
	ctxt, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	capacity := v1.ResourceList{}

	req := &runtimeapi.CapacityRequest{}

	resp, err := rgm.client.Capacity(ctxt, req)
	if err != nil {
		glog.V(2).Infof("Can not get capacity, %s", err)
		return capacity
	}

	gpus := resource.NewQuantity(int64(resp.Capacity), resource.DecimalSI)
	capacity[v1.ResourceNvidiaGPU] = *gpus

	return capacity
}

// AllocateGPUs returns `num` GPUs if available, error otherwise.
func (rgm *remoteGPUManager) AllocateGPU(pod *v1.Pod, container *v1.Container) ([]string, error) {
	rgm.Lock()
	defer rgm.Unlock()

	gpusNeeded := uint64(container.Resources.Limits.NvidiaGPU().Value())
	if gpusNeeded == 0 {
		return []string{}, nil
	}

	ctxt, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	podUID := string(pod.UID)
	req := &runtimeapi.AllocateGPURequest{
		RequestNum:    gpusNeeded,
		PodUid:        podUID,
		ContainerName: container.Name,
		Labels:        pod.Labels,
		Annotations:   pod.Annotations,
	}

	resp, err := rgm.client.AllocateGPU(ctxt, req)
	if err != nil {
		glog.V(2).Infof("Can not alllocate gpu, %s", err)
		return nil, err
	}

	var devices []string

	for _, item := range resp.GetDevices() {
		devices = append(devices, item.PathOnHost)
	}

	return devices, nil
}

func (rgm *remoteGPUManager) FreeGPU(pod *v1.Pod) {
	if pod == nil {
		return
	}

	ctxt, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	podUID := string(pod.UID)
	req := &runtimeapi.FreeGPURequest{
		PodUids: []string{podUID},
	}

	_, err := rgm.client.FreeGPU(ctxt, req)
	if err != nil {
		glog.V(5).Infof("Can not free gpu for removed pod: %s", podUID)
	}
}

func dial(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout("unix", addr, timeout)
}
