/*
Copyright 2015 Google Inc. All rights reserved.

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

package etcd

import (
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

type REST struct {
	*etcdgeneric.Etcd
	connection client.ConnectionInfoGetter
}

// NewStorage returns a RESTStorage object that will work against nodes.
func NewStorage(h tools.EtcdHelper, connection client.ConnectionInfoGetter) *REST {
	prefix := "/registry/minions"
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Node{} },
		NewListFunc: func() runtime.Object { return &api.NodeList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return prefix + "/" + name, nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Node).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return minion.MatchNode(label, field)
		},
		EndpointName: "minion",

		CreateStrategy: minion.Strategy,
		UpdateStrategy: minion.Strategy,

		Helper: h,
	}

	return &REST{store, connection}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified minion.
func (r *REST) ResourceLocation(ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	return minion.ResourceLocation(r, r.connection, ctx, id)
}
