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
	"io"
	"os/exec"
	"strconv"

	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
)

// GetContainerLogs uses journalctl to get the logs of the container.
// By default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
// TODO(yifan): Currently, it fetches all the containers' log within a pod. We will
// be able to fetch individual container's log once https://github.com/coreos/rkt/pull/841
// landed.
func (r *Runtime) GetContainerLogs(pod kubecontainer.Pod, tail string, follow bool, stdout, stderr io.Writer) error {
	unitName := makePodServiceFileName(pod.ID)
	cmd := exec.Command("journalctl", "-u", unitName)
	if follow {
		cmd.Args = append(cmd.Args, "-f")
	}
	if tail == "all" {
		cmd.Args = append(cmd.Args, "-a")
	} else {
		_, err := strconv.Atoi(tail)
		if err == nil {
			cmd.Args = append(cmd.Args, "-n", tail)
		}
	}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	return cmd.Start()
}
