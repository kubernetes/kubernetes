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

//go:build linux

package containerd

import (
	"context"
	"flag"
	"fmt"
	"path"
	"regexp"
	"strings"

	"k8s.io/klog/v2"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/watcher"
)

var ArgContainerdEndpoint = flag.String("containerd", "/run/containerd/containerd.sock", "containerd endpoint")
var ArgContainerdNamespace = flag.String("containerd-namespace", "k8s.io", "containerd namespace")

var containerdEnvMetadataWhiteList = flag.String("containerd_env_metadata_whitelist", "", "DEPRECATED: this flag will be removed, please use `env_metadata_whitelist`. A comma-separated list of environment variable keys matched with specified prefix that needs to be collected for containerd containers")

// The namespace under which containerd aliases are unique.
const k8sContainerdNamespace = "containerd"

// Regexp that identifies containerd cgroups, containers started with
// --cgroup-parent have another prefix than 'containerd'
var containerdCgroupRegexp = regexp.MustCompile(`([a-z0-9]{64})`)

type containerdFactory struct {
	machineInfoFactory info.MachineInfoFactory
	client             ContainerdClient
	version            string
	// Information about the mounted cgroup subsystems.
	cgroupSubsystems map[string]string
	// Information about mounted filesystems.
	fsInfo          fs.FsInfo
	includedMetrics container.MetricSet
}

func (f *containerdFactory) String() string {
	return k8sContainerdNamespace
}

func (f *containerdFactory) NewContainerHandler(name string, metadataEnvAllowList []string, inHostNamespace bool) (handler container.ContainerHandler, err error) {
	client, err := Client(*ArgContainerdEndpoint, *ArgContainerdNamespace)
	if err != nil {
		return
	}

	containerdMetadataEnvAllowList := strings.Split(*containerdEnvMetadataWhiteList, ",")

	// prefer using the unified metadataEnvAllowList
	if len(metadataEnvAllowList) != 0 {
		containerdMetadataEnvAllowList = metadataEnvAllowList
	}

	return newContainerdContainerHandler(
		client,
		name,
		f.machineInfoFactory,
		f.fsInfo,
		f.cgroupSubsystems,
		inHostNamespace,
		containerdMetadataEnvAllowList,
		f.includedMetrics,
	)
}

// Returns the containerd ID from the full container name.
func ContainerNameToContainerdID(name string) string {
	id := path.Base(name)
	if matches := containerdCgroupRegexp.FindStringSubmatch(id); matches != nil {
		return matches[1]
	}
	return id
}

// isContainerName returns true if the cgroup with associated name
// corresponds to a containerd container.
func isContainerName(name string) bool {
	// TODO: May be check with HasPrefix ContainerdNamespace
	if strings.HasSuffix(name, ".mount") {
		return false
	}
	return containerdCgroupRegexp.MatchString(path.Base(name))
}

// Containerd can handle and accept all containerd created containers
func (f *containerdFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	// if the container is not associated with containerd, we can't handle it or accept it.
	if !isContainerName(name) {
		return false, false, nil
	}
	// Check if the container is known to containerd and it is running.
	id := ContainerNameToContainerdID(name)
	// If container and task lookup in containerd fails then we assume
	// that the container state is not known to containerd
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()
	_, err := f.client.LoadContainer(ctx, id)
	if err != nil {
		return false, false, fmt.Errorf("failed to load container: %v", err)
	}

	return true, true, nil
}

func (f *containerdFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

// Register root container before running this function!
func Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics container.MetricSet) error {
	client, err := Client(*ArgContainerdEndpoint, *ArgContainerdNamespace)
	if err != nil {
		return fmt.Errorf("unable to create containerd client: %v", err)
	}

	containerdVersion, err := client.Version(context.Background())
	if err != nil {
		return fmt.Errorf("failed to fetch containerd client version: %v", err)
	}

	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems(includedMetrics)
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}

	klog.V(1).Infof("Registering containerd factory")
	f := &containerdFactory{
		cgroupSubsystems:   cgroupSubsystems,
		client:             client,
		fsInfo:             fsInfo,
		machineInfoFactory: factory,
		version:            containerdVersion,
		includedMetrics:    includedMetrics,
	}

	container.RegisterContainerHandlerFactory(f, []watcher.ContainerWatchSource{watcher.Raw})
	return nil
}
