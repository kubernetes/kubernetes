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
	"context"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newTargetProxyMetricContext(request string) *metricContext {
	return newGenericMetricContext("targetproxy", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetTargetHttpProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpProxy(name string) (*compute.TargetHttpProxy, error) {
	mc := newTargetProxyMetricContext("get")
	v, err := gce.c.TargetHttpProxies().Get(context.Background(), meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHttpProxy creates a TargetHttpProxy
func (gce *GCECloud) CreateTargetHttpProxy(proxy *compute.TargetHttpProxy) error {
	mc := newTargetProxyMetricContext("create")
	return mc.Observe(gce.c.TargetHttpProxies().Insert(context.Background(), meta.GlobalKey(proxy.Name), proxy))
}

// SetUrlMapForTargetHttpProxy sets the given UrlMap for the given TargetHttpProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpProxy(proxy *compute.TargetHttpProxy, urlMap *compute.UrlMap) error {
	ref := &compute.UrlMapReference{UrlMap: urlMap.SelfLink}
	mc := newTargetProxyMetricContext("set_url_map")
	return mc.Observe(gce.c.TargetHttpProxies().SetUrlMap(context.Background(), meta.GlobalKey(proxy.Name), ref))
}

// DeleteTargetHttpProxy deletes the TargetHttpProxy by name.
func (gce *GCECloud) DeleteTargetHttpProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(gce.c.TargetHttpProxies().Delete(context.Background(), meta.GlobalKey(name)))
}

// ListTargetHttpProxies lists all TargetHttpProxies in the project.
func (gce *GCECloud) ListTargetHttpProxies() ([]*compute.TargetHttpProxy, error) {
	mc := newTargetProxyMetricContext("list")
	v, err := gce.c.TargetHttpProxies().List(context.Background(), filter.None)
	return v, mc.Observe(err)
}

// TargetHttpsProxy management

// GetTargetHttpsProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpsProxy(name string) (*compute.TargetHttpsProxy, error) {
	mc := newTargetProxyMetricContext("get")
	v, err := gce.c.TargetHttpsProxies().Get(context.Background(), meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateTargetHttpsProxy creates a TargetHttpsProxy
func (gce *GCECloud) CreateTargetHttpsProxy(proxy *compute.TargetHttpsProxy) error {
	mc := newTargetProxyMetricContext("create")
	return mc.Observe(gce.c.TargetHttpsProxies().Insert(context.Background(), meta.GlobalKey(proxy.Name), proxy))
}

// SetUrlMapForTargetHttpsProxy sets the given UrlMap for the given TargetHttpsProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, urlMap *compute.UrlMap) error {
	mc := newTargetProxyMetricContext("set_url_map")
	ref := &compute.UrlMapReference{UrlMap: urlMap.SelfLink}
	return mc.Observe(gce.c.TargetHttpsProxies().SetUrlMap(context.Background(), meta.GlobalKey(proxy.Name), ref))
}

// SetSslCertificateForTargetHttpsProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (gce *GCECloud) SetSslCertificateForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, sslCert *compute.SslCertificate) error {
	mc := newTargetProxyMetricContext("set_ssl_cert")
	req := &compute.TargetHttpsProxiesSetSslCertificatesRequest{
		SslCertificates: []string{sslCert.SelfLink},
	}
	return mc.Observe(gce.c.TargetHttpsProxies().SetSslCertificates(context.Background(), meta.GlobalKey(proxy.Name), req))
}

// DeleteTargetHttpsProxy deletes the TargetHttpsProxy by name.
func (gce *GCECloud) DeleteTargetHttpsProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	return mc.Observe(gce.c.TargetHttpsProxies().Delete(context.Background(), meta.GlobalKey(name)))
}

// ListTargetHttpsProxies lists all TargetHttpsProxies in the project.
func (gce *GCECloud) ListTargetHttpsProxies() ([]*compute.TargetHttpsProxy, error) {
	mc := newTargetProxyMetricContext("list")
	v, err := gce.c.TargetHttpsProxies().List(context.Background(), filter.None)
	return v, mc.Observe(err)
}
