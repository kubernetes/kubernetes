/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

type HealthChecker interface {
	IsHealthy(container api.Container) (bool, error)
}

type httpDoInterface interface {
	Get(string) (*http.Response, error)
}

func MakeHealthChecker() HealthChecker {
	return &MuxHealthChecker{
		checkers: map[string]HealthChecker{
			"http": &HTTPHealthChecker{
				client: &http.Client{},
			},
		},
	}
}

type MuxHealthChecker struct {
	checkers map[string]HealthChecker
}

func (m *MuxHealthChecker) IsHealthy(container api.Container) (bool, error) {
	checker, ok := m.checkers[container.LivenessProbe.Type]
	if !ok || checker == nil {
		glog.Warningf("Failed to find health checker for %s %s", container.Name, container.LivenessProbe.Type)
		return true, nil
	}
	return checker.IsHealthy(container)
}

type HTTPHealthChecker struct {
	client httpDoInterface
}

func (h *HTTPHealthChecker) findPort(container api.Container, portName string) int64 {
	for _, port := range container.Ports {
		if port.Name == portName {
			// TODO This means you can only health check exposed ports
			return int64(port.HostPort)
		}
	}
	return -1
}

func (h *HTTPHealthChecker) IsHealthy(container api.Container) (bool, error) {
	params := container.LivenessProbe.HTTPGet
	port := h.findPort(container, params.Port)
	if port == -1 {
		var err error
		port, err = strconv.ParseInt(params.Port, 10, 0)
		if err != nil {
			return true, err
		}
	}
	var host string
	if len(params.Host) > 0 {
		host = params.Host
	} else {
		host = "localhost"
	}
	url := fmt.Sprintf("http://%s:%d%s", host, port, params.Path)
	res, err := h.client.Get(url)
	if res != nil && res.Body != nil {
		defer res.Body.Close()
	}
	if err != nil {
		// At this point, if it fails, its either a policy (unlikely) or HTTP protocol (likely) error.
		return false, nil
	}
	return res.StatusCode == http.StatusOK, nil
}
