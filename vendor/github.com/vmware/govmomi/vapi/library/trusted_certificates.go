/*
Copyright (c) 2022-2022 VMware, Inc. All Rights Reserved.

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

package library

import (
	"context"
	"net/http"
	"path"

	"github.com/vmware/govmomi/vapi/internal"
)

// TrustedCertificate contains a trusted certificate in Base64 encoded PEM format
type TrustedCertificate struct {
	Text string `json:"cert_text"`
}

// TrustedCertificateSummary contains a trusted certificate in Base64 encoded PEM format and its id
type TrustedCertificateSummary struct {
	TrustedCertificate
	ID string `json:"certificate"`
}

// ListTrustedCertificates retrieves all content library's trusted certificates
func (c *Manager) ListTrustedCertificates(ctx context.Context) ([]TrustedCertificateSummary, error) {
	url := c.Resource(internal.TrustedCertificatesPath)
	var res struct {
		Certificates []TrustedCertificateSummary `json:"certificates"`
	}
	err := c.Do(ctx, url.Request(http.MethodGet), &res)
	return res.Certificates, err
}

// GetTrustedCertificate retrieves a trusted certificate for a given certificate id
func (c *Manager) GetTrustedCertificate(ctx context.Context, id string) (*TrustedCertificate, error) {
	url := c.Resource(path.Join(internal.TrustedCertificatesPath, id))
	var res TrustedCertificate
	err := c.Do(ctx, url.Request(http.MethodGet), &res)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// CreateTrustedCertificate adds a certificate to content library trust store
func (c *Manager) CreateTrustedCertificate(ctx context.Context, cert string) error {
	url := c.Resource(internal.TrustedCertificatesPath)
	body := TrustedCertificate{Text: cert}
	return c.Do(ctx, url.Request(http.MethodPost, body), nil)
}

// DeleteTrustedCertificate deletes the trusted certificate from content library's trust store for the given id
func (c *Manager) DeleteTrustedCertificate(ctx context.Context, id string) error {
	url := c.Resource(path.Join(internal.TrustedCertificatesPath, id))
	return c.Do(ctx, url.Request(http.MethodDelete), nil)
}
