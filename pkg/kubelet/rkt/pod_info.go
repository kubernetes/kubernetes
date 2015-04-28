/*
Copyright 2015 Google Inc. All rights reserved.

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

package rkt

import (
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type podInfo struct {
	state       string
	networkInfo string
	pid         int
	// A map from image hashes to exit codes.
	// TODO(yifan): Should be appName to exit code in the future.
	exitCodes map[string]int
}

func newPodInfo() *podInfo {
	return &podInfo{pid: -1, exitCodes: make(map[string]int)}
}

func (p *podInfo) parseStatus(status []string) error {
	for _, line := range status {
		tuples := strings.SplitN(line, "=", 2)
		if len(tuples) != 2 {
			glog.Warningf("Invalid status line: %q", line)
			continue
		}
		switch tuples[0] {
		case "state":
			p.state = tuples[1]
		case "networks":
			p.networkInfo = tuples[1]
		case "pid":
			pid, err := strconv.Atoi(tuples[1])
			if err != nil {
				glog.Errorf("Cannot parse pid %q: %v", tuples[1], err)
				continue
			}
			p.pid = pid
		}
		if strings.HasPrefix(tuples[0], "sha512") {
			if p.exitCodes == nil {
				p.exitCodes = make(map[string]int)
			}
			exitcode, err := strconv.Atoi(tuples[1])
			if err != nil {
				glog.Errorf("Cannot parse exit code %q: %v", tuples[1], err)
			} else {
				p.exitCodes[tuples[0]] = exitcode
			}
		}
	}
	return nil
}

// getIP returns the IP of a pod by parsing the network info.
// The network info looks like this:
//
// default:ip4=172.16.28.3, database:ip4=172.16.28.42
//
func (p *podInfo) getIP() string {
	parts := strings.Split(p.networkInfo, ",")

	for _, part := range parts {
		if strings.HasPrefix(part, "default:") {
			return strings.Split(part, "=")[1]
		}
	}
	return ""
}

// getContainerStatus converts the rkt pod state to the api.containerStatus.
// TODO(yifan): Get more detailed info such as Image, ImageID, etc.
func (p *podInfo) getContainerStatus(container *kubecontainer.Container) api.ContainerStatus {
	var status api.ContainerStatus
	status.Name = container.Name
	status.Image = container.Image

	containerID, _ := parseContainerID(string(container.ID))
	status.ImageID = containerID.imageID

	switch p.state {
	case Running:
		// TODO(yifan): Get StartedAt.
		status.State = api.ContainerState{
			Running: &api.ContainerStateRunning{
				StartedAt: util.Unix(container.Created, 0),
			},
		}
	case Embryo, Preparing, Prepared:
		status.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
	case AbortedPrepare, Deleting, Exited, Garbage:
		status.State = api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode:  p.exitCodes[status.ImageID],
				StartedAt: util.Unix(container.Created, 0),
			},
		}
	default:
		glog.Warningf("Unknown pod state: %q", p.state)
	}
	return status
}

func (p *podInfo) toPodStatus(pod *kubecontainer.Pod) api.PodStatus {
	var status api.PodStatus
	status.PodIP = p.getIP()
	// For now just make every container's state as same as the pod.
	for _, container := range pod.Containers {
		status.ContainerStatuses = append(status.ContainerStatuses, p.getContainerStatus(container))
	}
	return status
}
