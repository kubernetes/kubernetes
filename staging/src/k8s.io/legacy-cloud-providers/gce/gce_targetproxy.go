//go:build !providerless
// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	compute "google.golang.org/api/compute/v1"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

func newTargetProxyMetricContext(request string) *metricContext {
	return newGenericMetricContext("targetproxy", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetTargetHTTPProxy returns the UrlMap by name.
func (g *Cloud) GetTargetHTTPProxy(name string) (*compute.TargetHttpProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("get")
	v, err := g.c.TargetHttpProxies().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHTTPProxy creates a TargetHttpProxy
func (g *Cloud) CreateTargetHTTPProxy(proxy *compute.TargetHttpProxy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("create")
	return mc.Observe(g.c.TargetHttpProxies().Insert(ctx, meta.GlobalKey(proxy.Name), proxy))
}

// SetURLMapForTargetHTTPProxy sets the given UrlMap for the given TargetHttpProxy.
func (g *Cloud) SetURLMapForTargetHTTPProxy(proxy *compute.TargetHttpProxy, urlMapLink string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	ref := &compute.UrlMapReference{UrlMap: urlMapLink}
	mc := newTargetProxyMetricContext("set_url_map")
	return mc.Observe(g.c.TargetHttpProxies().SetUrlMap(ctx, meta.GlobalKey(proxy.Name), ref))
}

// DeleteTargetHTTPProxy deletes the TargetHttpProxy by name.
func (g *Cloud) DeleteTargetHTTPProxy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(g.c.TargetHttpProxies().Delete(ctx, meta.GlobalKey(name)))
}

// ListTargetHTTPProxies lists all TargetHttpProxies in the project.
func (g *Cloud) ListTargetHTTPProxies() ([]*compute.TargetHttpProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("list")
	v, err := g.c.TargetHttpProxies().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// TargetHttpsProxy management

// GetTargetHTTPSProxy returns the UrlMap by name.
func (g *Cloud) GetTargetHTTPSProxy(name string) (*compute.TargetHttpsProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("get")
	v, err := g.c.TargetHttpsProxies().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHTTPSProxy creates a TargetHttpsProxy
func (g *Cloud) CreateTargetHTTPSProxy(proxy *compute.TargetHttpsProxy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("create")
	return mc.Observe(g.c.TargetHttpsProxies().Insert(ctx, meta.GlobalKey(proxy.Name), proxy))
}

// SetURLMapForTargetHTTPSProxy sets the given UrlMap for the given TargetHttpsProxy.
func (g *Cloud) SetURLMapForTargetHTTPSProxy(proxy *compute.TargetHttpsProxy, urlMapLink string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("set_url_map")
	ref := &compute.UrlMapReference{UrlMap: urlMapLink}
	return mc.Observe(g.c.TargetHttpsProxies().SetUrlMap(ctx, meta.GlobalKey(proxy.Name), ref))
}

// SetSslCertificateForTargetHTTPSProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (g *Cloud) SetSslCertificateForTargetHTTPSProxy(proxy *compute.TargetHttpsProxy, sslCertURLs []string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("set_ssl_cert")
	req := &compute.TargetHttpsProxiesSetSslCertificatesRequest{
		SslCertificates: sslCertURLs,
	}
	return mc.Observe(g.c.TargetHttpsProxies().SetSslCertificates(ctx, meta.GlobalKey(proxy.Name), req))
}

// DeleteTargetHTTPSProxy deletes the TargetHttpsProxy by name.
func (g *Cloud) DeleteTargetHTTPSProxy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(g.c.TargetHttpsProxies().Delete(ctx, meta.GlobalKey(name)))
}

// ListTargetHTTPSProxies lists all TargetHttpsProxies in the project.
func (g *Cloud) ListTargetHTTPSProxies() ([]*compute.TargetHttpsProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("list")
	v, err := g.c.TargetHttpsProxies().List(ctx, filter.None)
	return v, mc.Observe(err)
}
