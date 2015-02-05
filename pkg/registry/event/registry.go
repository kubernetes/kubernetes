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

package event

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// registry implements custom changes to generic.Etcd.
type registry struct {
	*etcdgeneric.Etcd
	ttl uint64
}

// Create stores the object with a ttl, so that events don't stay in the system forever.
func (r registry) Create(ctx api.Context, id string, obj runtime.Object) error {
	key, err := r.Etcd.KeyFunc(ctx, id)
	if err != nil {
		return err
	}
	err = r.Etcd.Helper.CreateObj(key, obj, r.ttl)
	return etcderr.InterpretCreateError(err, r.Etcd.EndpointName, id)
}

// Update replaces an existing instance of the object, and sets a ttl so that the event
// doesn't stay in the system forever.
func (r registry) Update(ctx api.Context, id string, obj runtime.Object) error {
	key, err := r.Etcd.KeyFunc(ctx, id)
	if err != nil {
		return err
	}
	err = r.Etcd.Helper.SetObj(key, obj, r.ttl)
	return etcderr.InterpretUpdateError(err, r.Etcd.EndpointName, id)
}

// NewEtcdRegistry returns a registry which will store Events in the given
// EtcdHelper. ttl is the time that Events will be retained by the system.
func NewEtcdRegistry(h tools.EtcdHelper, ttl uint64) generic.Registry {
	return registry{
		Etcd: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &api.Event{} },
			NewListFunc:  func() runtime.Object { return &api.EventList{} },
			EndpointName: "events",
			KeyRootFunc: func(ctx api.Context) string {
				return etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/events")
			},
			KeyFunc: func(ctx api.Context, id string) (string, error) {
				return etcdgeneric.NamespaceKeyFunc(ctx, "/registry/events", id)
			},
			Helper: h,
		},
		ttl: ttl,
	}
}
