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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
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

func (r *HealthyRegistry) GetMinion(ctx api.Context, minionID string) (*api.Node, error) {
	minion, err := r.delegate.GetMinion(ctx, minionID)
	if err != nil {
		return nil, err
	}
	return r.checkMinion(minion), nil
}

func (r *HealthyRegistry) DeleteMinion(ctx api.Context, minionID string) error {
	return r.delegate.DeleteMinion(ctx, minionID)
}

func (r *HealthyRegistry) CreateMinion(ctx api.Context, minion *api.Node) error {
	return r.delegate.CreateMinion(ctx, minion)
}

func (r *HealthyRegistry) UpdateMinion(ctx api.Context, minion *api.Node) error {
	return r.delegate.UpdateMinion(ctx, minion)
}

func (r *HealthyRegistry) ListMinions(ctx api.Context) (currentMinions *api.NodeList, err error) {
	list, err := r.delegate.ListMinions(ctx)
	if err != nil {
		return nil, err
	}
	for i := range list.Items {
		list.Items[i] = *r.checkMinion(&list.Items[i])
	}
	return list, nil
}

func (r *HealthyRegistry) WatchMinions(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	w, err := r.delegate.WatchMinions(ctx, label, field, resourceVersion)
	if err != nil {
		return nil, err
	}
	return watch.Filter(w, watch.FilterFunc(func(in watch.Event) (watch.Event, bool) {
		if node, ok := in.Object.(*api.Node); ok && node != nil {
			in.Object = r.checkMinion(node)
		}
		return in, true
	})), nil
}

func (r *HealthyRegistry) checkMinion(node *api.Node) *api.Node {
	condition := api.ConditionFull
	switch status, err := r.client.HealthCheck(node.Name); {
	case err != nil:
		condition = api.ConditionUnknown
	case status == health.Unhealthy:
		condition = api.ConditionNone
	}
	// TODO: distinguish other conditions like Reachable/Live, and begin storing this
	// data on nodes directly via sync loops.
	node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
		Kind:   api.NodeReady,
		Status: condition,
	})
	return node
}
