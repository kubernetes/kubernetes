/*
Copyright 2023 The Kubernetes Authors.

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

package ktesting

import (
	"fmt"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

// WithRESTConfig initializes all client-go clients with new clients
// created for the config. The current test name gets included in the UserAgent.
func (tc *TC) WithRESTConfig(cfg *rest.Config) TContext {
	cfg = rest.CopyConfig(cfg)
	cfg.UserAgent = fmt.Sprintf("%s -- %s", rest.DefaultKubernetesUserAgent(), tc.Name())

	tc = tc.clone()
	tc.restConfig = cfg
	tc.client = clientset.NewForConfigOrDie(cfg)
	tc.dynamic = dynamic.NewForConfigOrDie(cfg)
	tc.apiextensions = apiextensions.NewForConfigOrDie(cfg)
	cachedDiscovery := memory.NewMemCacheClient(tc.client.Discovery())
	tc.restMapper = restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	return tc
}

// WithClients uses an existing config and clients.
func (tc *TC) WithClients(cfg *rest.Config, mapper *restmapper.DeferredDiscoveryRESTMapper, client clientset.Interface, dynamic dynamic.Interface, apiextensions apiextensions.Interface) TContext {
	tc = tc.clone()
	tc.restConfig = cfg
	tc.restMapper = mapper
	tc.client = client
	tc.dynamic = dynamic
	tc.apiextensions = apiextensions
	return tc
}
