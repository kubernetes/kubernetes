/*
Copyright 2019 The Kubernetes Authors.

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

package renewal

import (
	"crypto/x509"
	"sort"

	"github.com/pkg/errors"
	clientset "k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// Manager can be used to coordinate certificate renewal and related processes,
// like CSR generation or checking certificate expiration
type Manager struct {
	// cfg holds the kubeadm ClusterConfiguration
	cfg *kubeadmapi.ClusterConfiguration

	// kubernetesDir holds the directory where kubeConfig files are stored
	kubernetesDir string

	// certificates contains the certificateRenewHandler controlled by this manager
	certificates map[string]*CertificateRenewHandler
}

// CertificateRenewHandler defines required info for renewing a certificate
type CertificateRenewHandler struct {
	// Name of the certificate to be used for UX.
	// This value can be used to trigger operations on this certificate
	Name string

	// LongName of the certificate to be used for UX
	LongName string

	// FileName defines the name (or the BaseName) of the certificate file
	FileName string

	// CABaseName define the base name for the CA that should be used for certificate renewal
	CABaseName string

	// readwriter define a CertificateReadWriter to be used for certificate renewal
	readwriter certificateReadWriter
}

// NewManager return a new certificate renewal manager ready for handling certificates in the cluster
func NewManager(cfg *kubeadmapi.ClusterConfiguration, kubernetesDir string) (*Manager, error) {
	rm := &Manager{
		cfg:           cfg,
		kubernetesDir: kubernetesDir,
		certificates:  map[string]*CertificateRenewHandler{},
	}

	// gets the list of certificates that are expected according to the current cluster configuration
	certListFunc := certsphase.GetDefaultCertList
	if cfg.Etcd.External != nil {
		certListFunc = certsphase.GetCertsWithoutEtcd
	}
	certTree, err := certListFunc().AsMap().CertTree()
	if err != nil {
		return nil, err
	}

	// create a CertificateRenewHandler for each signed certificate in the certificate tree;
	// NB. we are not offering support for renewing CAs; this would cause serious consequences
	for ca, certs := range certTree {
		for _, cert := range certs {
			// create a ReadWriter for certificates stored in the K8s local PKI
			pkiReadWriter := newPKICertificateReadWriter(rm.cfg.CertificatesDir, cert.BaseName)

			// adds the certificateRenewHandler.
			// PKI certificates are indexed by name, that is a well know constant defined
			// in the certsphase package and that can be reused across all the kubeadm codebase
			rm.certificates[cert.Name] = &CertificateRenewHandler{
				Name:       cert.Name,
				LongName:   cert.LongName,
				FileName:   cert.BaseName,
				CABaseName: ca.BaseName, //Nb. this is a path for etcd certs (they are stored in a subfolder)
				readwriter: pkiReadWriter,
			}
		}
	}

	// gets the list of certificates that should be considered for renewal
	kubeConfigs := []struct {
		longName string
		fileName string
	}{
		{
			longName: "certificate embedded in the kubeconfig file for the admin to use and for kubeadm itself",
			fileName: kubeadmconstants.AdminKubeConfigFileName,
		},
		{
			longName: "certificate embedded in the kubeconfig file for the controller manager to use",
			fileName: kubeadmconstants.ControllerManagerKubeConfigFileName,
		},
		{
			longName: "certificate embedded in the kubeconfig file for the scheduler manager to use",
			fileName: kubeadmconstants.SchedulerKubeConfigFileName,
		},
		//NB. we are excluding KubeletKubeConfig from renewal because management of this certificate is delegated to kubelet
	}

	// create a CertificateRenewHandler for each kubeConfig file
	for _, kubeConfig := range kubeConfigs {
		// create a ReadWriter for certificates embedded in kubeConfig files
		kubeConfigReadWriter := newKubeconfigReadWriter(kubernetesDir, kubeConfig.fileName)

		// adds the certificateRenewHandler.
		// Certificates embedded kubeConfig files in are indexed by fileName, that is a well know constant defined
		// in the kubeadm constants package and that can be reused across all the kubeadm codebase
		rm.certificates[kubeConfig.fileName] = &CertificateRenewHandler{
			Name:       kubeConfig.fileName, // we are using fileName as name, because there is nothing similar outside
			LongName:   kubeConfig.longName,
			FileName:   kubeConfig.fileName,
			CABaseName: kubeadmconstants.CACertAndKeyBaseName, // all certificates in kubeConfig files are signed by the Kubernetes CA
			readwriter: kubeConfigReadWriter,
		}
	}

	return rm, nil
}

// Certificates return the list of certificates controlled by this Manager
func (rm *Manager) Certificates() []*CertificateRenewHandler {
	certificates := []*CertificateRenewHandler{}
	for _, h := range rm.certificates {
		certificates = append(certificates, h)
	}

	sort.Slice(certificates, func(i, j int) bool { return certificates[i].Name < certificates[j].Name })

	return certificates
}

// RenewUsingLocalCA executes certificate renewal using local certificate authorities for generating new certs.
// For PKI certificates, use the name defined in the certsphase package, while for certificates
// embedded in the kubeConfig files, use the kubeConfig file name defined in the kubeadm constants package.
// If you use the CertificateRenewHandler returned by Certificates func, handler.Name already contains the right value.
func (rm *Manager) RenewUsingLocalCA(name string) (bool, error) {
	handler, ok := rm.certificates[name]
	if !ok {
		return false, errors.Errorf("%s is not a valid certificate for this cluster", name)
	}

	// checks if the certificate is externally managed (CA certificate provided without the certificate key)
	externallyManaged, err := rm.IsExternallyManaged(handler)
	if err != nil {
		return false, err
	}

	// in case of external CA it is not possible to renew certificates, then return early
	if externallyManaged {
		return false, nil
	}

	// reads the current certificate
	cert, err := handler.readwriter.Read()
	if err != nil {
		return false, err
	}

	// extract the certificate config
	cfg := certToConfig(cert)

	// reads the CA
	caCert, caKey, err := certsphase.LoadCertificateAuthority(rm.cfg.CertificatesDir, handler.CABaseName)
	if err != nil {
		return false, err
	}

	// create a new certificate with the same config
	newCert, newKey, err := NewFileRenewer(caCert, caKey).Renew(cfg)
	if err != nil {
		return false, errors.Wrapf(err, "failed to renew certificate %s", name)
	}

	// writes the new certificate to disk
	err = handler.readwriter.Write(newCert, newKey)
	if err != nil {
		return false, err
	}

	return true, nil
}

// RenewUsingCSRAPI executes certificate renewal uses the K8s certificate API.
// For PKI certificates, use the name defined in the certsphase package, while for certificates
// embedded in the kubeConfig files, use the kubeConfig file name defined in the kubeadm constants package.
// If you use the CertificateRenewHandler returned by Certificates func, handler.Name already contains the right value.
func (rm *Manager) RenewUsingCSRAPI(name string, client clientset.Interface) error {
	handler, ok := rm.certificates[name]
	if !ok {
		return errors.Errorf("%s is not a valid certificate for this cluster", name)
	}

	// reads the current certificate
	cert, err := handler.readwriter.Read()
	if err != nil {
		return err
	}

	// extract the certificate config
	cfg := certToConfig(cert)

	// create a new certificate with the same config
	newCert, newKey, err := NewAPIRenewer(client).Renew(cfg)
	if err != nil {
		return errors.Wrapf(err, "failed to renew certificate %s", name)
	}

	// writes the new certificate to disk
	err = handler.readwriter.Write(newCert, newKey)
	if err != nil {
		return err
	}

	return nil
}

// CreateRenewCSR generates CSR request for certificate renewal.
// For PKI certificates, use the name defined in the certsphase package, while for certificates
// embedded in the kubeConfig files, use the kubeConfig file name defined in the kubeadm constants package.
// If you use the CertificateRenewHandler returned by Certificates func, handler.Name already contains the right value.
func (rm *Manager) CreateRenewCSR(name, outdir string) error {
	handler, ok := rm.certificates[name]
	if !ok {
		return errors.Errorf("%s is not a known certificate", name)
	}

	// reads the current certificate
	cert, err := handler.readwriter.Read()
	if err != nil {
		return err
	}

	// extracts the certificate config
	cfg := certToConfig(cert)

	// generates the CSR request and save it
	csr, key, err := pkiutil.NewCSRAndKey(cfg)
	if err := pkiutil.WriteKey(outdir, name, key); err != nil {
		return errors.Wrapf(err, "failure while saving %s key", name)
	}

	if err := pkiutil.WriteCSR(outdir, name, csr); err != nil {
		return errors.Wrapf(err, "failure while saving %s CSR", name)
	}

	return nil
}

// GetExpirationInfo returns certificate expiration info.
// For PKI certificates, use the name defined in the certsphase package, while for certificates
// embedded in the kubeConfig files, use the kubeConfig file name defined in the kubeadm constants package.
// If you use the CertificateRenewHandler returned by Certificates func, handler.Name already contains the right value.
func (rm *Manager) GetExpirationInfo(name string) (*ExpirationInfo, error) {
	handler, ok := rm.certificates[name]
	if !ok {
		return nil, errors.Errorf("%s is not a known certificate", name)
	}

	// checks if the certificate is externally managed (CA certificate provided without the certificate key)
	externallyManaged, err := rm.IsExternallyManaged(handler)
	if err != nil {
		return nil, err
	}

	// reads the current certificate
	cert, err := handler.readwriter.Read()
	if err != nil {
		return nil, err
	}

	// returns the certificate expiration info
	return newExpirationInfo(name, cert, externallyManaged), nil
}

// IsExternallyManaged checks if we are in the external CA case (CA certificate provided without the certificate key)
func (rm *Manager) IsExternallyManaged(h *CertificateRenewHandler) (bool, error) {
	switch h.CABaseName {
	case kubeadmconstants.CACertAndKeyBaseName:
		externallyManaged, err := certsphase.UsingExternalCA(rm.cfg)
		if err != nil {
			return false, errors.Wrapf(err, "Error checking external CA condition for %s certificate authority", h.CABaseName)
		}
		return externallyManaged, nil
	case kubeadmconstants.FrontProxyCACertAndKeyBaseName:
		externallyManaged, err := certsphase.UsingExternalFrontProxyCA(rm.cfg)
		if err != nil {
			return false, errors.Wrapf(err, "Error checking external CA condition for %s certificate authority", h.CABaseName)
		}
		return externallyManaged, nil
	case kubeadmconstants.EtcdCACertAndKeyBaseName:
		return false, nil
	default:
		return false, errors.Errorf("unknown certificate authority %s", h.CABaseName)
	}
}

func certToConfig(cert *x509.Certificate) *certutil.Config {
	return &certutil.Config{
		CommonName:   cert.Subject.CommonName,
		Organization: cert.Subject.Organization,
		AltNames: certutil.AltNames{
			IPs:      cert.IPAddresses,
			DNSNames: cert.DNSNames,
		},
		Usages: cert.ExtKeyUsage,
	}
}
