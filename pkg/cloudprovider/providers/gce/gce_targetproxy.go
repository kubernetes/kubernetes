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

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
)

func newTargetProxyMetricContext(request string) *metricContext {
	return newGenericMetricContext("targetproxy", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetTargetHttpProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpProxy(name string) (*compute.TargetHttpProxy, error) {
	mc := newTargetProxyMetricContext("get")
	glog.V(4).Infof("TargetHttpProxies.Get(%s, %s): start", gce.projectID, name)
	v, err := gce.service.TargetHttpProxies.Get(gce.projectID, name).Do()
	glog.V(4).Infof("TargetHttpProxies.Get(%s, %s): end", gce.projectID, name)
	return v, mc.Observe(err)
}

// CreateTargetHttpProxy creates a TargetHttpProxy
func (gce *GCECloud) CreateTargetHttpProxy(proxy *compute.TargetHttpProxy) error {
	mc := newTargetProxyMetricContext("create")
	glog.V(4).Infof("TargetHttpProxies.Insert(%s, %v): start", gce.projectID, proxy)
	op, err := gce.service.TargetHttpProxies.Insert(gce.projectID, proxy).Do()
	glog.V(4).Infof("TargetHttpProxies.Insert(%s, %v): end", gce.projectID, proxy)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// SetUrlMapForTargetHttpProxy sets the given UrlMap for the given TargetHttpProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpProxy(proxy *compute.TargetHttpProxy, urlMap *compute.UrlMap) error {
	mc := newTargetProxyMetricContext("set_url_map")
	uMap := compute.UrlMapReference{UrlMap: urlMap.SelfLink}
	glog.V(4).Infof("TargetHttpProxies.SetUrlMap(%s, %s, %v): start", gce.projectID, proxy.Name, uMap)
	op, err := gce.service.TargetHttpProxies.SetUrlMap(gce.projectID, proxy.Name, &uMap).Do()
	glog.V(4).Infof("TargetHttpProxies.SetUrlMap(%s, %s, %v): end", gce.projectID, proxy.Name, uMap)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// DeleteTargetHttpProxy deletes the TargetHttpProxy by name.
func (gce *GCECloud) DeleteTargetHttpProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	glog.V(4).Infof("TargetHttpProxies.Delete(%s, %s): start", gce.projectID, name)
	op, err := gce.service.TargetHttpProxies.Delete(gce.projectID, name).Do()
	glog.V(4).Infof("TargetHttpProxies.Delete(%s, %s): end", gce.projectID, name)
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
	glog.V(4).Infof("TargetHttpProxies.List(%s): start", gce.projectID)
	v, err := gce.service.TargetHttpProxies.List(gce.projectID).Do()
	glog.V(4).Infof("TargetHttpProxies.List(%s): end", gce.projectID)
	return v, mc.Observe(err)
}

// TargetHttpsProxy management

// GetTargetHttpsProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpsProxy(name string) (*compute.TargetHttpsProxy, error) {
	mc := newTargetProxyMetricContext("get")
	glog.V(4).Infof("TargetHttpProxies.Get(%s, %s): start", gce.projectID, name)
	v, err := gce.service.TargetHttpsProxies.Get(gce.projectID, name).Do()
	glog.V(4).Infof("TargetHttpProxies.Get(%s, %s): end", gce.projectID, name)
	return v, mc.Observe(err)
}

// CreateTargetHttpsProxy creates a TargetHttpsProxy
func (gce *GCECloud) CreateTargetHttpsProxy(proxy *compute.TargetHttpsProxy) error {
	mc := newTargetProxyMetricContext("create")
	glog.V(4).Infof("TargetHttpProxies.Insert(%s, %v): start", gce.projectID, proxy)
	op, err := gce.service.TargetHttpsProxies.Insert(gce.projectID, proxy).Do()
	glog.V(4).Infof("TargetHttpProxies.Insert(%s, %v): end", gce.projectID, proxy)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// SetUrlMapForTargetHttpsProxy sets the given UrlMap for the given TargetHttpsProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, urlMap *compute.UrlMap) error {
	mc := newTargetProxyMetricContext("set_url_map")
	uMap := compute.UrlMapReference{UrlMap: urlMap.SelfLink}
	glog.V(4).Infof("TargetHttpProxies.SetUrlMap(%s, %s, %v): start", gce.projectID, proxy, uMap)
	op, err := gce.service.TargetHttpsProxies.SetUrlMap(gce.projectID, proxy.Name, &uMap).Do()
	glog.V(4).Infof("TargetHttpProxies.SetUrlMap(%s, %s, %v): end", gce.projectID, proxy, uMap)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// SetSslCertificateForTargetHttpsProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (gce *GCECloud) SetSslCertificateForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, sslCert *compute.SslCertificate) error {
	mc := newTargetProxyMetricContext("set_ssl_cert")
	request := compute.TargetHttpsProxiesSetSslCertificatesRequest{SslCertificates: []string{sslCert.SelfLink}}
	glog.V(4).Infof("TargetHttpProxies.SetSslCertificates(%s, %s, %v): start", gce.projectID, proxy, request)
	op, err := gce.service.TargetHttpsProxies.SetSslCertificates(gce.projectID, proxy.Name, &request).Do()
	glog.V(4).Infof("TargetHttpProxies.SetSslCertificates(%s, %s, %v): end", gce.projectID, proxy, request)
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOp(op, mc)
}

// DeleteTargetHttpsProxy deletes the TargetHttpsProxy by name.
func (gce *GCECloud) DeleteTargetHttpsProxy(name string) error {
	mc := newTargetProxyMetricContext("delete")
	glog.V(4).Infof("TargetHttpProxies.Delete(%s, %s): start", gce.projectID, name)
	op, err := gce.service.TargetHttpsProxies.Delete(gce.projectID, name).Do()
	glog.V(4).Infof("TargetHttpProxies.Delete(%s, %s): end", gce.projectID, name)
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
	glog.V(4).Infof("TargetHttpProxies.List(%s): start", gce.projectID)
	v, err := gce.service.TargetHttpsProxies.List(gce.projectID).Do()
	glog.V(4).Infof("TargetHttpProxies.List(%s): end", gce.projectID)
	return v, mc.Observe(err)
}
