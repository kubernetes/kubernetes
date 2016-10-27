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

package qingcloud

import (
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
)

const (
	checkSleepDuration = 3 * time.Second
	waitActiveTimeout  = 60 * time.Second
	pageLimt           = 100
)

// Make sure qingcloud instance hostname or override-hostname (if provided) is equal to InstanceId
// Recommended to use override-hostname
func nodeNameToInstanceId(name types.NodeName) string {
	return string(name)
}

func getNodePort(service *api.Service, port int32, protocol api.Protocol) (nodePort int32, found bool) {
	if service == nil {
		return
	}

	for _, servicePort := range service.Spec.Ports {
		if servicePort.Port == port && servicePort.Protocol == protocol {
			nodePort = servicePort.NodePort
			found = true
			return
		}
	}

	return
}
