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

func newCertMetricContext(request string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"cert_" + request, unusedMetricLabel, unusedMetricLabel},
	}
}

// GetSslCertificate returns the SslCertificate by name.
func (gce *GCECloud) GetSslCertificate(name string) (*compute.SslCertificate, error) {
	mc := newCertMetricContext("get")
	v, err := gce.service.SslCertificates.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// CreateSslCertificate creates and returns a SslCertificate.
func (gce *GCECloud) CreateSslCertificate(sslCerts *compute.SslCertificate) (*compute.SslCertificate, error) {
	mc := newCertMetricContext("create")
	op, err := gce.service.SslCertificates.Insert(gce.projectID, sslCerts).Do()

	if err != nil {
		return nil, mc.Observe(err)
	}

	if err = gce.waitForGlobalOp(op, mc); err != nil {
		return nil, mc.Observe(err)
	}

	return gce.GetSslCertificate(sslCerts.Name)
}

// DeleteSslCertificate deletes the SslCertificate by name.
func (gce *GCECloud) DeleteSslCertificate(name string) error {
	mc := newCertMetricContext("delete")
	op, err := gce.service.SslCertificates.Delete(gce.projectID, name).Do()

	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}

		return mc.Observe(err)
	}

	return gce.waitForGlobalOp(op, mc)
}

// ListSslCertificates lists all SslCertificates in the project.
func (gce *GCECloud) ListSslCertificates() (*compute.SslCertificateList, error) {
	mc := newCertMetricContext("list")
	// TODO: use PageToken to list all not just the first 500
	v, err := gce.service.SslCertificates.List(gce.projectID).Do()
	return v, mc.Observe(err)
}
