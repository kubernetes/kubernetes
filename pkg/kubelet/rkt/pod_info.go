/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util"
)

// rkt pod state.
// TODO(yifan): Use exported definition in rkt.
const (
	Embryo         = "embryo"
	Preparing      = "preparing"
	AbortedPrepare = "aborted prepare"
	Prepared       = "prepared"
	Running        = "running"
	Deleting       = "deleting" // This covers pod.isExitedDeleting and pod.isDeleting.
	Exited         = "exited"   // This covers pod.isExited and pod.isExitedGarbage.
	Garbage        = "garbage"

	// The prefix before the app name for each app's exit code in the output of 'rkt status'.
	exitCodePrefix = "app-"
)

// rktInfo represents the information of the rkt pod that stored in the
// systemd service file.
type rktInfo struct {
	uuid         string
	restartCount int
}

func emptyRktInfo() *rktInfo {
	return &rktInfo{restartCount: -1}
}

func (r *rktInfo) isEmpty() bool {
	return reflect.DeepEqual(r, emptyRktInfo())
}

// podInfo is the internal type that represents the state of
// the rkt pod.
type podInfo struct {
	// The state of the pod, e.g. Embryo, Preparing.
	state string
	// The ip of the pod. IPv4 for now.
	ip string
	// The pid of the init process in the pod.
	pid int
	// A map of [app name]:[exit code].
	exitCodes map[string]int
	// TODO(yifan): Expose [app name]:[image id].
}

// parsePodInfo parses the result of 'rkt status' into podInfo.
//
// Example output of 'rkt status':
//
// state=exited
// pid=-1
// exited=true
// app-etcd=0         # The exit code of the app "etcd" in the pod.
// app-redis=0        # The exit code of the app "redis" in the pod.
//
func parsePodInfo(status []string) (*podInfo, error) {
	p := &podInfo{
		pid:       -1,
		exitCodes: make(map[string]int),
	}

	for _, line := range status {
		tuples := strings.SplitN(line, "=", 2)
		if len(tuples) != 2 {
			return nil, fmt.Errorf("invalid status line: %q", line)
		}
		switch tuples[0] {
		case "state":
			// TODO(yifan): Parse the status here. This requires more details in
			// the rkt status, (e.g. started time, image name, etc).
			p.state = tuples[1]
		case "networks":
			p.ip = getIPFromNetworkInfo(tuples[1])
		case "pid":
			pid, err := strconv.Atoi(tuples[1])
			if err != nil {
				return nil, fmt.Errorf("cannot parse pid from %s: %v", tuples[1], err)
			}
			p.pid = pid
		}

		if strings.HasPrefix(tuples[0], exitCodePrefix) {
			exitcode, err := strconv.Atoi(tuples[1])
			if err != nil {
				return nil, fmt.Errorf("cannot parse exit code from %s : %v", tuples[1], err)
			}
			appName := strings.TrimPrefix(tuples[0], exitCodePrefix)
			p.exitCodes[appName] = exitcode
		}
	}
	return p, nil
}

// getIPFromNetworkInfo returns the IP of a pod by parsing the network info.
// The network info looks like this:
//
// default:ip4=172.16.28.3
// database:ip4=172.16.28.42
//
func getIPFromNetworkInfo(networkInfo string) string {
	parts := strings.Split(networkInfo, ",")
	for _, part := range parts {
		tuples := strings.Split(part, "=")
		if len(tuples) == 2 {
			return tuples[1]
		}
	}
	return ""
}

// makeContainerStatus creates the api.containerStatus of a container from the podInfo.
func makeContainerStatus(container *kubecontainer.Container, podInfo *podInfo) api.ContainerStatus {
	var status api.ContainerStatus
	status.Name = container.Name
	status.Image = container.Image
	status.ContainerID = string(container.ID)
	// TODO(yifan): Add image ID info.

	switch podInfo.state {
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
		exitCode, ok := podInfo.exitCodes[status.Name]
		if !ok {
			glog.Warningf("rkt: Cannot get exit code for container %v", container)
			exitCode = -1

		}
		status.State = api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode:  exitCode,
				StartedAt: util.Unix(container.Created, 0),
			},
		}
	default:
		glog.Warningf("rkt: Unknown pod state: %q", podInfo.state)
	}
	return status
}

// makePodStatus constructs the pod status from the pod info and rkt info.
func makePodStatus(pod *kubecontainer.Pod, podInfo *podInfo, rktInfo *rktInfo) api.PodStatus {
	var status api.PodStatus
	status.PodIP = podInfo.ip
	// For now just make every container's state the same as the pod.
	for _, container := range pod.Containers {
		containerStatus := makeContainerStatus(container, podInfo)
		containerStatus.RestartCount = rktInfo.restartCount
		status.ContainerStatuses = append(status.ContainerStatuses, containerStatus)
	}
	return status
}

// splitLineByTab breaks a line by tabs, and trims the leading and tailing spaces.
func splitLineByTab(line string) []string {
	var result []string
	tuples := strings.Split(strings.TrimSpace(line), "\t")
	for _, t := range tuples {
		if t != "" {
			result = append(result, t)
		}
	}
	return result
}
