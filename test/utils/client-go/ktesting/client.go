/*
Copyright The Kubernetes Authors.

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
	"context"
	"fmt"
	"testing"
	"time"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type data struct {
	restConfig    *rest.Config
	restMapper    *restmapper.DeferredDiscoveryRESTMapper
	client        clientset.Interface
	dynamic       dynamic.Interface
	apiextensions apiextensions.Interface
}

type dataKeyType struct{}

var dataKey dataKeyType

func get(ctx context.Context) data {
	c := ctx.Value(dataKey)
	if c == nil {
		return data{}
	}
	return *c.(*data)
}

func set(tCtx ktesting.TContext, data data) TContext {
	return TContext{tCtx.WithValue(dataKey, &data)}
}

// TContext is a drop-in replacement for [ktesting.TContext]. It adds support
// for client-go.
//
// New methods allow retrieving instances of several client-go types without
// having to pass all of them down into call chains of test helper functions as
// separate parameters.
//
// All methods which have a [ktesting.TContext] in their prototype get
// overridden such that they work with a TContext instead. This ensures
// that the new methods are always accessible.
//
// When calling functions which expect a [ktesting.TContext] pass
// the TContext field embedded here.
type TContext struct {
	ktesting.TContext
}

// RESTConfig returns a copy of the config for a rest client with the UserAgent
// set to include the current test name or nil if not available. Several typed
// clients using this config are available through [Client], [Dynamic],
// [APIExtensions].
func (tCtx TContext) RESTConfig() *rest.Config {
	return rest.CopyConfig(get(tCtx).restConfig)
}

func (tCtx TContext) RESTMapper() *restmapper.DeferredDiscoveryRESTMapper {
	return get(tCtx).restMapper
}
func (tCtx TContext) Client() clientset.Interface            { return get(tCtx).client }
func (tCtx TContext) Dynamic() dynamic.Interface             { return get(tCtx).dynamic }
func (tCtx TContext) APIExtensions() apiextensions.Interface { return get(tCtx).apiextensions }

// WithRESTConfig initializes all client-go clients with new clients
// created for the config. The current test name gets included in the UserAgent.
func WithRESTConfig(tCtx ktesting.TContext, cfg *rest.Config) TContext {
	cfg = rest.CopyConfig(cfg)
	cfg.UserAgent = fmt.Sprintf("%s -- %s", rest.DefaultKubernetesUserAgent(), tCtx.Name())

	var data data
	data.restConfig = cfg
	data.client = clientset.NewForConfigOrDie(cfg)
	data.dynamic = dynamic.NewForConfigOrDie(cfg)
	data.apiextensions = apiextensions.NewForConfigOrDie(cfg)
	cachedDiscovery := memory.NewMemCacheClient(data.client.Discovery())
	data.restMapper = restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	return set(tCtx, data)
}

func (tCtx TContext) WithRESTConfig(cfg *rest.Config) TContext {
	return WithRESTConfig(tCtx.TContext, cfg)
}

// WithClients uses an existing config and clients.
func WithClients(tCtx ktesting.TContext, cfg *rest.Config, mapper *restmapper.DeferredDiscoveryRESTMapper, client clientset.Interface, dynamic dynamic.Interface, apiextensions apiextensions.Interface) TContext {
	return set(tCtx, data{
		restConfig:    cfg,
		restMapper:    mapper,
		client:        client,
		dynamic:       dynamic,
		apiextensions: apiextensions,
	})
}

func (tCtx TContext) WithClients(cfg *rest.Config, mapper *restmapper.DeferredDiscoveryRESTMapper, client clientset.Interface, dynamic dynamic.Interface, apiextensions apiextensions.Interface) TContext {
	return WithClients(tCtx.TContext, cfg, mapper, client, dynamic, apiextensions)
}

// See [ktesting.TB].
type TB = ktesting.TB

// See [ktesting.ContextTB].
type ContextTB = ktesting.ContextTB

// See [ktesting.WithStep].
func (tCtx TContext) WithStep(step string) TContext {
	return TContext{tCtx.TContext.WithStep(step)}
}

// See [ktesting.Step].
func (tCtx TContext) Step(step string, cb func(tCtx TContext)) {
	tCtx.TContext.Step(step, func(tCtx ktesting.TContext) {
		cb(TContext{tCtx})
	})
}

// See [ktesting.Init].
func Init(tb ktesting.TB, opts ...ktesting.InitOption) TContext {
	return TContext{ktesting.Init(tb, opts...)}
}

// See [ktesting.InitCtx].
func InitCtx(ctx context.Context, tb ktesting.TB, opts ...ktesting.InitOption) TContext {
	return TContext{ktesting.InitCtx(ctx, tb, opts...)}
}

// See [ktesting.WithContext].
func (tCtx TContext) WithContext(ctx context.Context) TContext {
	return TContext{tCtx.TContext.WithContext(ctx)}
}

// See [ktesting.WithValue].
func (tCtx TContext) WithValue(key, val any) TContext {
	return TContext{tCtx.TContext.WithValue(key, val)}
}

// See [ktesting.WithNamespace].
func (tCtx TContext) WithNamespace(namespace string) TContext {
	return TContext{tCtx.TContext.WithNamespace(namespace)}
}

// See [ktesting.WithCancel].
func (tCtx TContext) WithCancel() TContext {
	return TContext{tCtx.TContext.WithCancel()}
}

// See [ktesting.WithoutCancel].
func (tCtx TContext) WithoutCancel() TContext {
	return TContext{tCtx.TContext.WithoutCancel()}
}

// See [ktesting.WithTimeout].
func (tCtx TContext) WithTimeout(timeout time.Duration, timeoutCause string) TContext {
	return TContext{tCtx.TContext.WithTimeout(timeout, timeoutCause)}
}

// See [ktesting.WithLogger].
func (tCtx TContext) WithLogger(logger klog.Logger) TContext {
	return TContext{tCtx.TContext.WithLogger(logger)}
}

// See [ktesting.CleanupCtx].
func (tCtx TContext) CleanupCtx(cb func(tCtx TContext)) {
	tCtx.Helper()
	tCtx.TContext.CleanupCtx(func(tCtx ktesting.TContext) {
		cb(TContext{tCtx})
	})
}

// See [ktesting.WithError].
func (tCtx TContext) WithError(err *error) (TContext, func()) {
	baseCtx, finalize := tCtx.TContext.WithError(err)
	return TContext{baseCtx}, finalize
}

// See [ktesting.Run].
func (tCtx TContext) Run(name string, cb func(tCtx TContext)) bool {
	return tCtx.TContext.Run(name, func(tCtx ktesting.TContext) {
		cb(TContext{tCtx})
	})
}

// See [ktesting.SyncTest].
func (tCtx TContext) SyncTest(name string, cb func(tCtx TContext)) bool {
	return tCtx.TContext.SyncTest(name, func(tCtx ktesting.TContext) {
		cb(TContext{tCtx})
	})
}

// See [ktesting.SetDefaultVerbosity].
func SetDefaultVerbosity(v int) {
	ktesting.SetDefaultVerbosity(v)
}

// See [ktesting.NewTestContext].
func NewTestContext(tb testing.TB) (klog.Logger, context.Context) {
	logger, ctx := ktesting.NewTestContext(tb)
	return logger, TContext{ctx.(ktesting.TContext)}
}
