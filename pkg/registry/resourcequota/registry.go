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

package resourcequota

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequotausage"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// Registry implements operations to modify ResourceQuota objects
type Registry interface {
	generic.Registry
	resourcequotausage.Registry
}

// registry implements custom changes to generic.Etcd.
type registry struct {
	*etcdgeneric.Etcd
}

// ApplyStatus atomically updates the ResourceQuotaStatus based on the observed ResourceQuotaUsage
func (r *registry) ApplyStatus(ctx api.Context, usage *api.ResourceQuotaUsage) error {
	obj, err := r.Get(ctx, usage.Name)
	if err != nil {
		return err
	}

	if len(usage.ResourceVersion) == 0 {
		return fmt.Errorf("A resource observation must have a resourceVersion specified to ensure atomic updates")
	}

	// set the status
	resourceQuota := obj.(*api.ResourceQuota)
	resourceQuota.ResourceVersion = usage.ResourceVersion
	resourceQuota.Status = usage.Status
	return r.UpdateWithName(ctx, resourceQuota.Name, resourceQuota)
}

// NewEtcdRegistry returns a registry which will store ResourceQuota in the given helper
func NewEtcdRegistry(h tools.EtcdHelper) Registry {
	return &registry{
		Etcd: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &api.ResourceQuota{} },
			NewListFunc:  func() runtime.Object { return &api.ResourceQuotaList{} },
			EndpointName: "resourcequotas",
			KeyRootFunc: func(ctx api.Context) string {
				return etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/resourcequotas")
			},
			KeyFunc: func(ctx api.Context, id string) (string, error) {
				return etcdgeneric.NamespaceKeyFunc(ctx, "/registry/resourcequotas", id)
			},
			Helper: h,
		},
	}
}
