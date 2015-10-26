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
	"strings"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// containerID defines the ID of rkt containers, it will
// be returned to kubelet, and kubelet will use this for
// container level operations.
type containerID struct {
	uuid    string // rkt uuid of the pod.
	appName string // Name of the app in that pod.
}

const RktType = "rkt"

// buildContainerID constructs the containers's ID using containerID,
// which consists of the pod uuid and the container name.
// The result can be used to uniquely identify a container.
func buildContainerID(c *containerID) kubecontainer.ContainerID {
	return kubecontainer.ContainerID{
		Type: RktType,
		ID:   fmt.Sprintf("%s:%s", c.uuid, c.appName),
	}
}

// parseContainerID parses the containerID into pod uuid and the container name. The
// results can be used to get more information of the container.
func parseContainerID(id kubecontainer.ContainerID) (*containerID, error) {
	tuples := strings.Split(id.ID, ":")
	if len(tuples) != 2 {
		return nil, fmt.Errorf("rkt: cannot parse container ID for: %v", id)
	}
	return &containerID{
		uuid:    tuples[0],
		appName: tuples[1],
	}, nil
}
