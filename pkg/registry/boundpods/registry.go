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

package boundpods

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Registry is the BoundPods storage interface
type Registry interface {
	Get(ctx api.Context, id string) (runtime.Object, error)
	Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// etcd implements a registry for bound pods that extends generic.Etcd.
type etcdRegistry struct {
	generic *etcdgeneric.Etcd
}

// NewEtcdRegistry returns a registry which will retrieve BoundPods per node with the
// provided EtcdHelper.
func NewEtcdRegistry(h tools.EtcdHelper) Registry {
	return &etcdRegistry{
		generic: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &api.BoundPods{} },
			NewListFunc:  func() runtime.Object { return nil },
			EndpointName: "boundPods",
			KeyRootFunc: func(_ api.Context) string {
				return "/registry/nodes"
			},
			KeyFunc: func(_ api.Context, id string) (string, error) {
				return fmt.Sprintf("/registry/nodes/%s/boundpods", id), nil
			},
			Helper: h,
		},
	}
}

// Get implements Registry and returns a matching BoundPods object
func (e *etcdRegistry) Get(ctx api.Context, id string) (runtime.Object, error) {
	return e.generic.Get(ctx, id)
}

// hostField is used to identify the field name that may be watched on for bound pods
const hostField = "host"

// Watch starts a watch for the items that have matching labels and fields.
func (e *etcdRegistry) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	version, err := tools.ParseWatchResourceVersion(resourceVersion, e.generic.EndpointName)
	if err != nil {
		return nil, err
	}
	if !label.Empty() {
		return nil, fmt.Errorf("selecting on labels for bound pods is not supported")
	}
	if host, ok := field.RequiresExactMatch(hostField); ok {
		key, err := e.generic.KeyFunc(ctx, host)
		if err != nil {
			return nil, err
		}
		return e.generic.Helper.Watch(key, version), nil
	}
	if !field.Empty() {
		return nil, fmt.Errorf("only the %q field may be used to filter a watch on bound pods", hostField)
	}
	return e.generic.Helper.WatchList(e.generic.KeyRootFunc(ctx), version, func(obj runtime.Object) bool {
		_, ok := obj.(*api.BoundPods)
		return ok
	})
}
