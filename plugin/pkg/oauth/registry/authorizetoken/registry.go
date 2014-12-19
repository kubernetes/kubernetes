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

package authorizetoken

import (
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

// Registry is an interface for things that know how to store AuthorizeToken objects.
type Registry interface {
	generic.Registry
}

type registry struct {
	*etcdgeneric.Etcd
}

// NewEtcdRegistry returns a registry which will store AuthorizeTokens in the given EtcdHelper
func NewEtcdRegistry(h tools.EtcdHelper) Registry {
	return registry{
		Etcd: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &oapi.OAuthAuthorizeToken{} },
			NewListFunc:  func() runtime.Object { return &oapi.OAuthAuthorizeTokenList{} },
			EndpointName: "oAuthAuthorizeTokens",
			KeyRootFunc: func(ctx api.Context) string {
				return "/registry/oauth/authorizeTokens"
			},
			KeyFunc: func(ctx api.Context, name string) (string, error) {
				return path.Join("/registry/oauth/authorizeTokens", name), nil
			},
			Helper: h,
		},
	}
}
