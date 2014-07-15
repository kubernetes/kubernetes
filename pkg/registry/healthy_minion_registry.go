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

package registry

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
)

type HealthyMinionRegistry struct {
	delegate MinionRegistry
	client   health.HTTPGetInterface
	port     int
}

func NewHealthyMinionRegistry(delegate MinionRegistry, client *http.Client) MinionRegistry {
	return &HealthyMinionRegistry{
		delegate: delegate,
		client:   client,
		port:     10250,
	}
}

func (h *HealthyMinionRegistry) makeMinionURL(minion string) string {
	return fmt.Sprintf("http://%s:%d/healthz", minion, h.port)
}

func (h *HealthyMinionRegistry) List() (currentMinions []string, err error) {
	var result []string
	list, err := h.delegate.List()
	if err != nil {
		return result, err
	}
	for _, minion := range list {
		status, err := health.Check(h.makeMinionURL(minion), h.client)
		if err != nil {
			return result, err
		}
		if status == health.Healthy {
			result = append(result, minion)
		}
	}
	return result, nil
}

func (h *HealthyMinionRegistry) Insert(minion string) error {
	return h.delegate.Insert(minion)
}

func (h *HealthyMinionRegistry) Delete(minion string) error {
	return h.delegate.Delete(minion)
}

func (h *HealthyMinionRegistry) Contains(minion string) (bool, error) {
	contains, err := h.delegate.Contains(minion)
	if err != nil {
		return false, err
	}
	if !contains {
		return false, nil
	}
	status, err := health.Check(h.makeMinionURL(minion), h.client)
	if err != nil {
		return false, err
	}
	if status == health.Unhealthy {
		return false, nil
	}
	return true, nil
}
