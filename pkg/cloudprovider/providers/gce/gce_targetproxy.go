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
	"net/http"
	"time"

	compute "google.golang.org/api/compute/v1"
)

func newTargetProxyMetricContext(request string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"targetproxy_" + request, unusedMetricLabel, unusedMetricLabel},
	}
}

// GetTargetHttpProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpProxy(name string) (*compute.TargetHttpProxy, error) {
	mc := newTargetProxyMetricContext("get")
	v, err := gce.service.TargetHttpProxies.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// CreateTargetHttpProxy creates and returns a TargetHttpProxy with the given UrlMap.
func (gce *GCECloud) CreateTargetHttpProxy(urlMap *compute.UrlMap, name string) (*compute.TargetHttpProxy, error) {
	proxy := &compute.TargetHttpProxy{
		Name:   name,
		UrlMap: urlMap.SelfLink,
	}

	mc := newTargetProxyMetricContext("create")
	op, err := gce.service.TargetHttpProxies.Insert(gce.projectID, proxy).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	if err = gce.waitForGlobalOp(op, mc); err != nil {
		return nil, err
	}
	return gce.GetTargetHttpProxy(name)
}

// SetUrlMapForTargetHttpProxy sets the given UrlMap for the given TargetHttpProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpProxy(proxy *compute.TargetHttpProxy, urlMap *compute.UrlMap) error {
	mc := newTargetProxyMetricContext("set_url_map")
	op, err := gce.service.TargetHttpProxies.SetUrlMap(
		gce.projectID, proxy.Name, &compute.UrlMapReference{UrlMap: urlMap.SelfLink}).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// DeleteTargetHttpProxy deletes the TargetHttpProxy by name.
func (gce *GCECloud) DeleteTargetHttpProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	op, err := gce.service.TargetHttpProxies.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// ListTargetHttpProxies lists all TargetHttpProxies in the project.
func (gce *GCECloud) ListTargetHttpProxies() (*compute.TargetHttpProxyList, error) {
	mc := newTargetProxyMetricContext("list")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.TargetHttpProxies.List(gce.projectID).Do()
	return v, mc.Observe(err)
}

// TargetHttpsProxy management

// GetTargetHttpsProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpsProxy(name string) (*compute.TargetHttpsProxy, error) {
	mc := newTargetProxyMetricContext("get")
	v, err := gce.service.TargetHttpsProxies.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// CreateTargetHttpsProxy creates and returns a TargetHttpsProxy with the given UrlMap and SslCertificate.
func (gce *GCECloud) CreateTargetHttpsProxy(urlMap *compute.UrlMap, sslCert *compute.SslCertificate, name string) (*compute.TargetHttpsProxy, error) {
	mc := newTargetProxyMetricContext("create")
	proxy := &compute.TargetHttpsProxy{
		Name:            name,
		UrlMap:          urlMap.SelfLink,
		SslCertificates: []string{sslCert.SelfLink},
	}
	op, err := gce.service.TargetHttpsProxies.Insert(gce.projectID, proxy).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	if err = gce.waitForGlobalOp(op, mc); err != nil {
		return nil, err
	}
	return gce.GetTargetHttpsProxy(name)
}

// SetUrlMapForTargetHttpsProxy sets the given UrlMap for the given TargetHttpsProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, urlMap *compute.UrlMap) error {
	mc := newTargetProxyMetricContext("set_url_map")
	op, err := gce.service.TargetHttpsProxies.SetUrlMap(
		gce.projectID, proxy.Name, &compute.UrlMapReference{UrlMap: urlMap.SelfLink}).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// SetSslCertificateForTargetHttpsProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (gce *GCECloud) SetSslCertificateForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, sslCert *compute.SslCertificate) error {
	mc := newTargetProxyMetricContext("set_ssl_cert")
	op, err := gce.service.TargetHttpsProxies.SetSslCertificates(
		gce.projectID, proxy.Name, &compute.TargetHttpsProxiesSetSslCertificatesRequest{SslCertificates: []string{sslCert.SelfLink}}).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// DeleteTargetHttpsProxy deletes the TargetHttpsProxy by name.
func (gce *GCECloud) DeleteTargetHttpsProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	op, err := gce.service.TargetHttpsProxies.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// ListTargetHttpsProxies lists all TargetHttpsProxies in the project.
func (gce *GCECloud) ListTargetHttpsProxies() (*compute.TargetHttpsProxyList, error) {
	mc := newTargetProxyMetricContext("list")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.TargetHttpsProxies.List(gce.projectID).Do()
	return v, mc.Observe(err)
}
