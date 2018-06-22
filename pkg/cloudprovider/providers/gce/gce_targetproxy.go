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

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newTargetProxyMetricContext(request string) *metricContext {
	return newGenericMetricContext("targetproxy", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetTargetHttpProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpProxy(name string) (*compute.TargetHttpProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("get")
	v, err := gce.c.TargetHttpProxies().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHttpProxy creates a TargetHttpProxy
func (gce *GCECloud) CreateTargetHttpProxy(proxy *compute.TargetHttpProxy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("create")
	return mc.Observe(gce.c.TargetHttpProxies().Insert(ctx, meta.GlobalKey(proxy.Name), proxy))
}

// SetUrlMapForTargetHttpProxy sets the given UrlMap for the given TargetHttpProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpProxy(proxy *compute.TargetHttpProxy, urlMapLink string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	ref := &compute.UrlMapReference{UrlMap: urlMapLink}
	mc := newTargetProxyMetricContext("set_url_map")
	return mc.Observe(gce.c.TargetHttpProxies().SetUrlMap(ctx, meta.GlobalKey(proxy.Name), ref))
}

// DeleteTargetHttpProxy deletes the TargetHttpProxy by name.
func (gce *GCECloud) DeleteTargetHttpProxy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(gce.c.TargetHttpProxies().Delete(ctx, meta.GlobalKey(name)))
}

// ListTargetHttpProxies lists all TargetHttpProxies in the project.
func (gce *GCECloud) ListTargetHttpProxies() ([]*compute.TargetHttpProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("list")
	v, err := gce.c.TargetHttpProxies().List(ctx, filter.None)
	return v, mc.Observe(err)
}

// TargetHttpsProxy management

// GetTargetHttpsProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpsProxy(name string) (*compute.TargetHttpsProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("get")
	v, err := gce.c.TargetHttpsProxies().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHttpsProxy creates a TargetHttpsProxy
func (gce *GCECloud) CreateTargetHttpsProxy(proxy *compute.TargetHttpsProxy) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("create")
	return mc.Observe(gce.c.TargetHttpsProxies().Insert(ctx, meta.GlobalKey(proxy.Name), proxy))
}

// SetUrlMapForTargetHttpsProxy sets the given UrlMap for the given TargetHttpsProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, urlMapLink string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("set_url_map")
	ref := &compute.UrlMapReference{UrlMap: urlMapLink}
	return mc.Observe(gce.c.TargetHttpsProxies().SetUrlMap(ctx, meta.GlobalKey(proxy.Name), ref))
}

// SetSslCertificateForTargetHttpsProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (gce *GCECloud) SetSslCertificateForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, sslCertURLs []string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("set_ssl_cert")
	req := &compute.TargetHttpsProxiesSetSslCertificatesRequest{
		SslCertificates: sslCertURLs,
	}
	return mc.Observe(gce.c.TargetHttpsProxies().SetSslCertificates(ctx, meta.GlobalKey(proxy.Name), req))
}

// DeleteTargetHttpsProxy deletes the TargetHttpsProxy by name.
func (gce *GCECloud) DeleteTargetHttpsProxy(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(gce.c.TargetHttpsProxies().Delete(ctx, meta.GlobalKey(name)))
}

// ListTargetHttpsProxies lists all TargetHttpsProxies in the project.
func (gce *GCECloud) ListTargetHttpsProxies() ([]*compute.TargetHttpsProxy, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newTargetProxyMetricContext("list")
	v, err := gce.c.TargetHttpsProxies().List(ctx, filter.None)
	return v, mc.Observe(err)
}
