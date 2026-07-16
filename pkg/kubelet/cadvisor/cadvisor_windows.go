//go:build windows

/*
Copyright 2015 The Kubernetes Authors.

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

package cadvisor

import (
	"context"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

type cadvisorClient struct {
	rootPath       string
	winStatsClient winstats.Client
}

var _ Interface = new(cadvisorClient)

// New creates a cAdvisor and exports its API on the specified port if port > 0.
func New(logger klog.Logger, imageFsInfoProvider ImageFsInfoProvider, rootPath string, cgroupRoots []string, usingLegacyStats, localStorageCapacityIsolation bool) (Interface, error) {
	client, err := winstats.NewPerfCounterClient(logger)
	return &cadvisorClient{
		rootPath:       rootPath,
		winStatsClient: client,
	}, err
}

func (cu *cadvisorClient) Start() error {
	return nil
}

// ContainerInfoV2 is only expected to be used for the root container. Returns info for all containers in the node.
func (cu *cadvisorClient) ContainerInfoV2(name string, options cadvisorapi.RequestOptions) (map[string]cadvisorapi.ContainerInfo, error) {
	return cu.winStatsClient.WinContainerInfos()
}

func (cu *cadvisorClient) GetRequestedContainersInfo(containerName string, options cadvisorapi.RequestOptions) (map[string]*cadvisorapi.ContainerInfo, error) {
	return nil, nil
}

func (cu *cadvisorClient) MachineInfo(logger klog.Logger) (*cadvisorapi.MachineInfo, error) {
	return cu.winStatsClient.WinMachineInfo(logger)
}

func (cu *cadvisorClient) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return cu.winStatsClient.WinVersionInfo()
}

func (cu *cadvisorClient) ImagesFsInfo(context.Context) (cadvisorapi.FsInfo, error) {
	return cadvisorapi.FsInfo{}, nil
}

func (cu *cadvisorClient) ContainerFsInfo(context.Context) (cadvisorapi.FsInfo, error) {
	return cadvisorapi.FsInfo{}, nil
}

func (cu *cadvisorClient) RootFsInfo() (cadvisorapi.FsInfo, error) {
	return cu.GetDirFsInfo(cu.rootPath)
}

func (cu *cadvisorClient) GetDirFsInfo(path string) (cadvisorapi.FsInfo, error) {
	return cu.winStatsClient.GetDirFsInfo(path)
}

func IsPsiEnabled(_ klog.Logger) bool {
	return false
}
