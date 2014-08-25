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

package minion

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"

	"github.com/golang/glog"
)

type HealthyRegistry struct {
	delegate Registry
	client   health.HTTPGetInterface
	port     int
}

func NewHealthyRegistry(delegate Registry, client *http.Client) Registry {
	return &HealthyRegistry{
		delegate: delegate,
		client:   client,
		port:     10250,
	}
}

func (r *HealthyRegistry) Contains(minion string) (bool, error) {
	contains, err := r.delegate.Contains(minion)
	if err != nil {
		return false, err
	}
	if !contains {
		return false, nil
	}
	status, err := health.DoHTTPCheck(r.makeMinionURL(minion), r.client)
	if err != nil {
		return false, err
	}
	if status == health.Unhealthy {
		return false, nil
	}
	return true, nil
}

func (r *HealthyRegistry) Delete(minion string) error {
	return r.delegate.Delete(minion)
}

func (r *HealthyRegistry) Insert(minion string) error {
	return r.delegate.Insert(minion)
}

func (r *HealthyRegistry) List() (currentMinions []string, err error) {
	var result []string
	list, err := r.delegate.List()
	if err != nil {
		return result, err
	}
	for _, minion := range list {
		status, err := health.DoHTTPCheck(r.makeMinionURL(minion), r.client)
		if err != nil {
			glog.Errorf("%s failed health check with error: %s", minion, err)
			continue
		}
		if status == health.Healthy {
			result = append(result, minion)
		} else {
			glog.Errorf("%s failed a health check, ignoring.", minion)
		}
	}
	return result, nil
}

func (r *HealthyRegistry) makeMinionURL(minion string) string {
	return fmt.Sprintf("http://%s:%d/healthz", minion, r.port)
}
