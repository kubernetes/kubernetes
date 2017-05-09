/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package collector

import (
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type ContainerInfoData interface{}

type Interface interface {
	// Starts the collector process if necessary, e.g. when using the builtin cadvisor.
	// Otherwise, does nothing and just returns.
	Start() error

	// Returns the basic node spec, e.g. can be used to report to k8s master the
	// basic information about a node.
	NodeInfo() (*NodeInfo, error)

	// Returns the software version of various components of a node, e.g. can be
	// used to report back to k8s master the versions of relevant software on a node.
	VersionInfo() (*VersionInfo, error)

	// Returns the file system information of a given label or mount point (mount point
	// is represented in absolute path, i.e. starting with '/', and label is not), e.g.
	// can be used by kubelet to check if there's enough disk space to schedule new pods
	// and by the image manager to decide if garbage collection should start.
	FsInfo(fsLabel string) (*FsInfo, error)

	// Returns a channel for watching requested event types, e.g. can be
	// used by oom watcher to detect oom events.
	WatchEvents(request *EventRequest) (chan *Event, error)

	// Get raw metrics of a Docker/Rkt container
	// containerID: Docker container ID
	// numStats: max number of metrics to return
	// start: start time to query the metrics data, and if omitted, the beginning of time is assumed
	// end: end time to query the metrics data, and if omitted, the current time is assumed
	// Returned value is a map of container name to its metrics data, and this data can be in a
	// collector-specific format, e.g. can be used to handle /stats/... REST calls
	ContainerInfo(containerID kubecontainer.ContainerID, numStats int, start time.Time, end time.Time) (ContainerInfoData, error)

	// Get raw metrics of a Linux container
	// containerName: absolute path of a Linux container
	// numStats: max number of metrics to return
	// start: start time to query the metrics data, and if omitted, the beginning of time is assumed
	// end: end time to query the metrics data, and if omitted, the current time is assumed
	// subcontainer: if true, returns raw metrics information of this container and its child containers
	RawContainerInfo(containerName string, numStats int, start time.Time, end time.Time, subcontainers bool) (map[string]ContainerInfoData, error)

	// DockerContainerInfo() and RawContainerInfo() return collector-specific raw data, which need to be
	// parsed to retrieve a particular metric type, e.g. cpu, memory, cpu load, file system. The respective
	// parsing interface functions, e.g. GetContainerCPUInfo(), GetContainerMemoryInfo(), GetContainerNetworkInfo(),
	// etc., should be defined to retrieve such data in a format consumeable by Kubernetes. However, as of now
	// Kubernetes is currently not using any of such metrics data, thus, these interface functions (along with the
	// corresponding data structs) should be defined and tuned at the time when such code is added.

	// TODO: Define parsing interface functions as use cases are more concrete
}
