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

package masterflags

import (
	"path"

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
}

func (r registry) Create(ctx api.Context, id string, obj runtime.Object) error {
	err := r.Etcd.Helper.CreateObj(r.Etcd.KeyFunc(id), obj, 0)
	return etcderr.InterpretCreateError(err, r.Etcd.EndpointName, id)
}

// NewEtcdRegistry returns a registry which will store MasterFlags in the given
// EtcdHelper.
func NewEtcdRegistry(h tools.EtcdHelper) generic.Registry {
	return registry{
		Etcd: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &api.MasterFlags{} },
			NewListFunc:  func() runtime.Object { return &api.MasterFlagsList{} },
			EndpointName: "masterFlags",
			KeyRoot:      "/registry/masterFlags",
			KeyFunc: func(id string) string {
				return path.Join("/registry/masterFlags", id)
			},
			Helper: h,
		},
	}
}
