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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"

	"github.com/golang/glog"
)

type HealthyRegistry struct {
	delegate Registry
	client   client.KubeletHealthChecker
}

func NewHealthyRegistry(delegate Registry, client client.KubeletHealthChecker) Registry {
	return &HealthyRegistry{
		delegate: delegate,
		client:   client,
	}
}

func (r *HealthyRegistry) GetMinion(ctx api.Context, minionID string) (*api.Minion, error) {
	minion, err := r.delegate.GetMinion(ctx, minionID)
	if minion == nil {
		return nil, ErrDoesNotExist
	}
	if err != nil {
		return nil, err
	}
	status, err := r.client.HealthCheck(minionID)
	if err != nil {
		return nil, err
	}
	if status == health.Unhealthy {
		return nil, ErrNotHealty
	}
	return minion, nil
}

func (r *HealthyRegistry) DeleteMinion(ctx api.Context, minionID string) error {
	return r.delegate.DeleteMinion(ctx, minionID)
}

func (r *HealthyRegistry) CreateMinion(ctx api.Context, minion *api.Minion) error {
	return r.delegate.CreateMinion(ctx, minion)
}

func (r *HealthyRegistry) ListMinions(ctx api.Context) (currentMinions *api.MinionList, err error) {
	result := &api.MinionList{}
	list, err := r.delegate.ListMinions(ctx)
	if err != nil {
		return result, err
	}
	for _, minion := range list.Items {
		status, err := r.client.HealthCheck(minion.Name)
		if err != nil {
			glog.V(1).Infof("%#v failed health check with error: %s", minion, err)
			continue
		}
		if status == health.Healthy {
			result.Items = append(result.Items, minion)
		} else {
			glog.Errorf("%s failed a health check, ignoring.", minion)
		}
	}
	return result, nil
}
