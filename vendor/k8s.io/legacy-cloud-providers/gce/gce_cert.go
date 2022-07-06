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

func newCertMetricContext(request string) *metricContext {
	return newGenericMetricContext("cert", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetSslCertificate returns the SslCertificate by name.
func (g *Cloud) GetSslCertificate(name string) (*compute.SslCertificate, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newCertMetricContext("get")
	v, err := g.c.SslCertificates().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateSslCertificate creates and returns a SslCertificate.
func (g *Cloud) CreateSslCertificate(sslCerts *compute.SslCertificate) (*compute.SslCertificate, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newCertMetricContext("create")
	err := g.c.SslCertificates().Insert(ctx, meta.GlobalKey(sslCerts.Name), sslCerts)
	if err != nil {
		return nil, mc.Observe(err)
	}
	return g.GetSslCertificate(sslCerts.Name)
}

// DeleteSslCertificate deletes the SslCertificate by name.
func (g *Cloud) DeleteSslCertificate(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newCertMetricContext("delete")
	return mc.Observe(g.c.SslCertificates().Delete(ctx, meta.GlobalKey(name)))
}

// ListSslCertificates lists all SslCertificates in the project.
func (g *Cloud) ListSslCertificates() ([]*compute.SslCertificate, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newCertMetricContext("list")
	v, err := g.c.SslCertificates().List(ctx, filter.None)
	return v, mc.Observe(err)
}
