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

	"github.com/onsi/gomega"
	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
)

// WithRESTConfig initializes all client-go clients with new clients
// created for the config. The current test name gets included in the UserAgent.
func WithRESTConfig(tCtx TContext, cfg *rest.Config) TContext {
	cfg = rest.CopyConfig(cfg)
	cfg.UserAgent = fmt.Sprintf("%s -- %s", rest.DefaultKubernetesUserAgent(), tCtx.Name())

	cCtx := clientContext{
		TContext:      tCtx,
		restConfig:    cfg,
		client:        clientset.NewForConfigOrDie(cfg),
		dynamic:       dynamic.NewForConfigOrDie(cfg),
		apiextensions: apiextensions.NewForConfigOrDie(cfg),
	}

	cachedDiscovery := memory.NewMemCacheClient(cCtx.client.Discovery())
	cCtx.restMapper = restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)

	return &cCtx
}

// WithClients uses an existing config and clients.
func WithClients(tCtx TContext, cfg *rest.Config, mapper *restmapper.DeferredDiscoveryRESTMapper, client clientset.Interface, dynamic dynamic.Interface, apiextensions apiextensions.Interface) TContext {
	return clientContext{
		TContext:      tCtx,
		restConfig:    cfg,
		restMapper:    mapper,
		client:        client,
		dynamic:       dynamic,
		apiextensions: apiextensions,
	}
}

type clientContext struct {
	TContext

	restConfig    *rest.Config
	restMapper    *restmapper.DeferredDiscoveryRESTMapper
	client        clientset.Interface
	dynamic       dynamic.Interface
	apiextensions apiextensions.Interface
}

func (cCtx clientContext) CleanupCtx(cb func(TContext)) {
	cCtx.Helper()
	cleanupCtx(cCtx, cb)
}

func (cCtx clientContext) Expect(actual interface{}, extra ...interface{}) gomega.Assertion {
	cCtx.Helper()
	return expect(cCtx, actual, extra...)
}

func (cCtx clientContext) ExpectNoError(err error, explain ...interface{}) {
	cCtx.Helper()
	expectNoError(cCtx, err, explain...)
}

func (cCtx clientContext) Run(name string, cb func(tCtx TContext)) bool {
	return run(cCtx, name, false, cb)
}

func (cCtx clientContext) SyncTest(name string, cb func(tCtx TContext)) bool {
	return run(cCtx, name, true, cb)
}

func (cCtx clientContext) Logger() klog.Logger {
	return klog.FromContext(cCtx)
}

func (cCtx clientContext) RESTConfig() *rest.Config {
	if cCtx.restConfig == nil {
		return nil
	}
	return rest.CopyConfig(cCtx.restConfig)
}

func (cCtx clientContext) RESTMapper() *restmapper.DeferredDiscoveryRESTMapper {
	return cCtx.restMapper
}

func (cCtx clientContext) Client() clientset.Interface {
	return cCtx.client
}

func (cCtx clientContext) Dynamic() dynamic.Interface {
	return cCtx.dynamic
}

func (cCtx clientContext) APIExtensions() apiextensions.Interface {
	return cCtx.apiextensions
}
