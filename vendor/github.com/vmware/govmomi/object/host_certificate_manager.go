/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// HostCertificateManager provides helper methods around the HostSystem.ConfigManager.CertificateManager
type HostCertificateManager struct {
	Common
	Host *HostSystem
}

// NewHostCertificateManager creates a new HostCertificateManager helper
func NewHostCertificateManager(c *vim25.Client, ref types.ManagedObjectReference, host types.ManagedObjectReference) *HostCertificateManager {
	return &HostCertificateManager{
		Common: NewCommon(c, ref),
		Host:   NewHostSystem(c, host),
	}
}

// CertificateInfo wraps the host CertificateManager certificateInfo property with the HostCertificateInfo helper.
// The ThumbprintSHA1 field is set to HostSystem.Summary.Config.SslThumbprint if the host system is managed by a vCenter.
func (m HostCertificateManager) CertificateInfo(ctx context.Context) (*HostCertificateInfo, error) {
	var hs mo.HostSystem
	var cm mo.HostCertificateManager

	pc := property.DefaultCollector(m.Client())

	err := pc.RetrieveOne(ctx, m.Reference(), []string{"certificateInfo"}, &cm)
	if err != nil {
		return nil, err
	}

	_ = pc.RetrieveOne(ctx, m.Host.Reference(), []string{"summary.config.sslThumbprint"}, &hs)

	return &HostCertificateInfo{
		HostCertificateManagerCertificateInfo: cm.CertificateInfo,
		ThumbprintSHA1:                        hs.Summary.Config.SslThumbprint,
	}, nil
}

// GenerateCertificateSigningRequest requests the host system to generate a certificate-signing request (CSR) for itself.
// The CSR is then typically provided to a Certificate Authority to sign and issue the SSL certificate for the host system.
// Use InstallServerCertificate to import this certificate.
func (m HostCertificateManager) GenerateCertificateSigningRequest(ctx context.Context, useIPAddressAsCommonName bool) (string, error) {
	req := types.GenerateCertificateSigningRequest{
		This:                     m.Reference(),
		UseIpAddressAsCommonName: useIPAddressAsCommonName,
	}

	res, err := methods.GenerateCertificateSigningRequest(ctx, m.Client(), &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

// GenerateCertificateSigningRequestByDn requests the host system to generate a certificate-signing request (CSR) for itself.
// Alternative version similar to GenerateCertificateSigningRequest but takes a Distinguished Name (DN) as a parameter.
func (m HostCertificateManager) GenerateCertificateSigningRequestByDn(ctx context.Context, distinguishedName string) (string, error) {
	req := types.GenerateCertificateSigningRequestByDn{
		This:              m.Reference(),
		DistinguishedName: distinguishedName,
	}

	res, err := methods.GenerateCertificateSigningRequestByDn(ctx, m.Client(), &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

// InstallServerCertificate imports the given SSL certificate to the host system.
func (m HostCertificateManager) InstallServerCertificate(ctx context.Context, cert string) error {
	req := types.InstallServerCertificate{
		This: m.Reference(),
		Cert: cert,
	}

	_, err := methods.InstallServerCertificate(ctx, m.Client(), &req)
	if err != nil {
		return err
	}

	// NotifyAffectedService is internal, not exposing as we don't have a use case other than with InstallServerCertificate
	// Without this call, hostd needs to be restarted to use the updated certificate
	// Note: using Refresh as it has the same struct/signature, we just need to use different xml name tags
	body := struct {
		Req *types.Refresh         `xml:"urn:vim25 NotifyAffectedServices,omitempty"`
		Res *types.RefreshResponse `xml:"urn:vim25 NotifyAffectedServicesResponse,omitempty"`
		methods.RefreshBody
	}{
		Req: &types.Refresh{This: m.Reference()},
	}

	return m.Client().RoundTrip(ctx, &body, &body)
}

// ListCACertificateRevocationLists returns the SSL CRLs of Certificate Authorities that are trusted by the host system.
func (m HostCertificateManager) ListCACertificateRevocationLists(ctx context.Context) ([]string, error) {
	req := types.ListCACertificateRevocationLists{
		This: m.Reference(),
	}

	res, err := methods.ListCACertificateRevocationLists(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

// ListCACertificates returns the SSL certificates of Certificate Authorities that are trusted by the host system.
func (m HostCertificateManager) ListCACertificates(ctx context.Context) ([]string, error) {
	req := types.ListCACertificates{
		This: m.Reference(),
	}

	res, err := methods.ListCACertificates(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

// ReplaceCACertificatesAndCRLs replaces the trusted CA certificates and CRL used by the host system.
// These determine whether the server can verify the identity of an external entity.
func (m HostCertificateManager) ReplaceCACertificatesAndCRLs(ctx context.Context, caCert []string, caCrl []string) error {
	req := types.ReplaceCACertificatesAndCRLs{
		This:   m.Reference(),
		CaCert: caCert,
		CaCrl:  caCrl,
	}

	_, err := methods.ReplaceCACertificatesAndCRLs(ctx, m.Client(), &req)
	return err
}
