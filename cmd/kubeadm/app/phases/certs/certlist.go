/*
Copyright 2018 The Kubernetes Authors.

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

package certs

import (
	"crypto"
	"crypto/x509"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/pkg/errors"

	certutil "k8s.io/client-go/util/cert"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

const (
	errInvalid = "invalid argument"
	errExist   = "file already exists"
)

type configMutatorsFunc func(*kubeadmapi.InitConfiguration, *pkiutil.CertConfig) error

// KubeadmCert represents a certificate that Kubeadm will create to function properly.
type KubeadmCert struct {
	Name     string
	LongName string
	BaseName string
	CAName   string
	// Some attributes will depend on the InitConfiguration, only known at runtime.
	// These functions will be run in series, passed both the InitConfiguration and a cert Config.
	configMutators []configMutatorsFunc
	config         pkiutil.CertConfig
	// Used for unit tests.
	creationTime time.Time
}

// GetConfig returns the definition for the given cert given the provided InitConfiguration
func (k *KubeadmCert) GetConfig(ic *kubeadmapi.InitConfiguration) (*pkiutil.CertConfig, error) {
	for _, f := range k.configMutators {
		if err := f(ic, &k.config); err != nil {
			return nil, err
		}
	}

	// creationTime should be set only during unit tests, otherwise the kubeadm start time
	// should be
	if k.creationTime.IsZero() {
		k.creationTime = kubeadmutil.StartTimeUTC()
	}

	// Backdate certificate to allow small time jumps.
	k.config.NotBefore = k.creationTime.Add(-kubeadmconstants.CertificateBackdate)

	// Use the validity periods defined in the ClusterConfiguration.
	// If CAName is empty this is a CA cert.
	if len(k.CAName) != 0 {
		if ic.ClusterConfiguration.CertificateValidityPeriod != nil {
			k.config.NotAfter = k.creationTime.
				Add(ic.ClusterConfiguration.CertificateValidityPeriod.Duration)
		}
	} else {
		if ic.ClusterConfiguration.CACertificateValidityPeriod != nil {
			k.config.NotAfter = k.creationTime.
				Add(ic.ClusterConfiguration.CACertificateValidityPeriod.Duration)
		}
	}

	// Use the encryption algorithm from ClusterConfiguration.
	k.config.EncryptionAlgorithm = ic.ClusterConfiguration.EncryptionAlgorithmType()
	return &k.config, nil
}

// CreateFromCA makes and writes a certificate using the given CA cert and key.
func (k *KubeadmCert) CreateFromCA(ic *kubeadmapi.InitConfiguration, caCert *x509.Certificate, caKey crypto.Signer) error {
	cfg, err := k.GetConfig(ic)
	if err != nil {
		return errors.Wrapf(err, "couldn't create %q certificate", k.Name)
	}
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, cfg)
	if err != nil {
		return err
	}
	err = writeCertificateFilesIfNotExist(
		ic.CertificatesDir,
		k.BaseName,
		caCert,
		cert,
		key,
		cfg,
	)

	if err != nil {
		return errors.Wrapf(err, "failed to write or validate certificate %q", k.Name)
	}

	return nil
}

// CreateAsCA creates a certificate authority, writing the files to disk and also returning the created CA so it can be used to sign child certs.
func (k *KubeadmCert) CreateAsCA(ic *kubeadmapi.InitConfiguration) (*x509.Certificate, crypto.Signer, error) {
	cfg, err := k.GetConfig(ic)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't get configuration for %q CA certificate", k.Name)
	}
	caCert, caKey, err := pkiutil.NewCertificateAuthority(cfg)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't generate %q CA certificate", k.Name)
	}

	err = writeCertificateAuthorityFilesIfNotExist(
		ic.CertificatesDir,
		k.BaseName,
		caCert,
		caKey,
	)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't write out %q CA certificate", k.Name)
	}

	return caCert, caKey, nil
}

// CertificateTree is represents a one-level-deep tree, mapping a CA to the certs that depend on it.
type CertificateTree map[*KubeadmCert]Certificates

// CreateTree creates the CAs, certs signed by the CAs, and writes them all to disk.
func (t CertificateTree) CreateTree(ic *kubeadmapi.InitConfiguration) error {
	for ca, leaves := range t {
		cfg, err := ca.GetConfig(ic)
		if err != nil {
			return err
		}

		var caKey crypto.Signer

		caCert, err := pkiutil.TryLoadCertFromDisk(ic.CertificatesDir, ca.BaseName)
		if err == nil {
			// Validate period
			CheckCertificatePeriodValidity(ca.BaseName, caCert)

			// Cert exists already, make sure it's valid
			if !caCert.IsCA {
				return errors.Errorf("certificate %q is not a CA", ca.Name)
			}
			// Try and load a CA Key
			caKey, err = pkiutil.TryLoadKeyFromDisk(ic.CertificatesDir, ca.BaseName)
			if err != nil {
				// If there's no CA key, make sure every certificate exists.
				for _, leaf := range leaves {
					cl := certKeyLocation{
						pkiDir:   ic.CertificatesDir,
						baseName: leaf.BaseName,
						uxName:   leaf.Name,
					}
					if err := validateSignedCertWithCA(cl, caCert); err != nil {
						return errors.Wrapf(err, "could not load expected certificate %q or validate the existence of key %q for it", leaf.Name, ca.Name)
					}
				}
				continue
			}
			// CA key exists; just use that to create new certificates.
			klog.V(1).Infof("[certs] Using the existing CA certificate %q and key %q\n", filepath.Join(ic.CertificatesDir, fmt.Sprintf("%s.crt", ca.BaseName)), filepath.Join(ic.CertificatesDir, fmt.Sprintf("%s.key", ca.BaseName)))
		} else {
			// CACert doesn't already exist, create a new cert and key.
			caCert, caKey, err = pkiutil.NewCertificateAuthority(cfg)
			if err != nil {
				return err
			}

			err = writeCertificateAuthorityFilesIfNotExist(
				ic.CertificatesDir,
				ca.BaseName,
				caCert,
				caKey,
			)
			if err != nil {
				return err
			}
		}

		for _, leaf := range leaves {
			if err := leaf.CreateFromCA(ic, caCert, caKey); err != nil {
				return err
			}
		}
	}
	return nil
}

// CertificateMap is a flat map of certificates, keyed by Name.
type CertificateMap map[string]*KubeadmCert

// CertTree returns a one-level-deep tree, mapping a CA cert to an array of certificates that should be signed by it.
func (m CertificateMap) CertTree() (CertificateTree, error) {
	caMap := make(CertificateTree)

	for _, cert := range m {
		if cert.CAName == "" {
			if _, ok := caMap[cert]; !ok {
				caMap[cert] = []*KubeadmCert{}
			}
		} else {
			ca, ok := m[cert.CAName]
			if !ok {
				return nil, errors.Errorf("certificate %q references unknown CA %q", cert.Name, cert.CAName)
			}
			caMap[ca] = append(caMap[ca], cert)
		}
	}

	return caMap, nil
}

// Certificates is a list of Certificates that Kubeadm should create.
type Certificates []*KubeadmCert

// AsMap returns the list of certificates as a map, keyed by name.
func (c Certificates) AsMap() CertificateMap {
	certMap := make(map[string]*KubeadmCert)
	for _, cert := range c {
		certMap[cert.Name] = cert
	}

	return certMap
}

// GetDefaultCertList returns  all of the certificates kubeadm requires to function.
func GetDefaultCertList() Certificates {
	return Certificates{
		KubeadmCertRootCA(),
		KubeadmCertAPIServer(),
		KubeadmCertKubeletClient(),
		// Front Proxy certs
		KubeadmCertFrontProxyCA(),
		KubeadmCertFrontProxyClient(),
		// etcd certs
		KubeadmCertEtcdCA(),
		KubeadmCertEtcdServer(),
		KubeadmCertEtcdPeer(),
		KubeadmCertEtcdHealthcheck(),
		KubeadmCertEtcdAPIClient(),
	}
}

// GetCertsWithoutEtcd returns all of the certificates kubeadm needs when etcd is hosted externally.
func GetCertsWithoutEtcd() Certificates {
	return Certificates{
		KubeadmCertRootCA(),
		KubeadmCertAPIServer(),
		KubeadmCertKubeletClient(),
		// Front Proxy certs
		KubeadmCertFrontProxyCA(),
		KubeadmCertFrontProxyClient(),
	}
}

// KubeadmCertRootCA is the definition of the Kubernetes Root CA for the API Server and kubelet.
func KubeadmCertRootCA() *KubeadmCert {
	return &KubeadmCert{
		Name:     "ca",
		LongName: "self-signed Kubernetes CA to provision identities for other Kubernetes components",
		BaseName: kubeadmconstants.CACertAndKeyBaseName,
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: "kubernetes",
			},
		},
	}
}

// KubeadmCertAPIServer is the definition of the cert used to serve the Kubernetes API.
func KubeadmCertAPIServer() *KubeadmCert {
	return &KubeadmCert{
		Name:     "apiserver",
		LongName: "certificate for serving the Kubernetes API",
		BaseName: kubeadmconstants.APIServerCertAndKeyBaseName,
		CAName:   "ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: kubeadmconstants.APIServerCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
			},
		},
		configMutators: []configMutatorsFunc{
			makeAltNamesMutator(pkiutil.GetAPIServerAltNames),
		},
	}
}

// KubeadmCertKubeletClient is the definition of the cert used by the API server to access the kubelet.
func KubeadmCertKubeletClient() *KubeadmCert {
	return &KubeadmCert{
		Name:     "apiserver-kubelet-client",
		LongName: "certificate for the API server to connect to kubelet",
		BaseName: kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
		CAName:   "ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName:   kubeadmconstants.APIServerKubeletClientCertCommonName,
				Organization: []string{kubeadmconstants.ClusterAdminsGroupAndClusterRoleBinding},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// KubeadmCertFrontProxyCA is the definition of the CA used for the front end proxy.
func KubeadmCertFrontProxyCA() *KubeadmCert {
	return &KubeadmCert{
		Name:     "front-proxy-ca",
		LongName: "self-signed CA to provision identities for front proxy",
		BaseName: kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: "front-proxy-ca",
			},
		},
	}
}

// KubeadmCertFrontProxyClient is the definition of the cert used by the API server to access the front proxy.
func KubeadmCertFrontProxyClient() *KubeadmCert {
	return &KubeadmCert{
		Name:     "front-proxy-client",
		BaseName: kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
		LongName: "certificate for the front proxy client",
		CAName:   "front-proxy-ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: kubeadmconstants.FrontProxyClientCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// KubeadmCertEtcdCA is the definition of the root CA used by the hosted etcd server.
func KubeadmCertEtcdCA() *KubeadmCert {
	return &KubeadmCert{
		Name:     "etcd-ca",
		LongName: "self-signed CA to provision identities for etcd",
		BaseName: kubeadmconstants.EtcdCACertAndKeyBaseName,
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: "etcd-ca",
			},
		},
	}
}

// KubeadmCertEtcdServer is the definition of the cert used to serve etcd to clients.
func KubeadmCertEtcdServer() *KubeadmCert {
	return &KubeadmCert{
		Name:     "etcd-server",
		LongName: "certificate for serving etcd",
		BaseName: kubeadmconstants.EtcdServerCertAndKeyBaseName,
		CAName:   "etcd-ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				// TODO: etcd 3.2 introduced an undocumented requirement for ClientAuth usage on the
				// server cert: https://github.com/etcd-io/etcd/issues/9785#issuecomment-396715692
				// Once the upstream issue is resolved, this should be returned to only allowing
				// ServerAuth usage.
				Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
			},
		},
		configMutators: []configMutatorsFunc{
			makeAltNamesMutator(pkiutil.GetEtcdAltNames),
			setCommonNameToNodeName(),
		},
	}
}

// KubeadmCertEtcdPeer is the definition of the cert used by etcd peers to access each other.
func KubeadmCertEtcdPeer() *KubeadmCert {
	return &KubeadmCert{
		Name:     "etcd-peer",
		LongName: "certificate for etcd nodes to communicate with each other",
		BaseName: kubeadmconstants.EtcdPeerCertAndKeyBaseName,
		CAName:   "etcd-ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
			},
		},
		configMutators: []configMutatorsFunc{
			makeAltNamesMutator(pkiutil.GetEtcdPeerAltNames),
			setCommonNameToNodeName(),
		},
	}
}

// KubeadmCertEtcdHealthcheck is the definition of the cert used by Kubernetes to check the health of the etcd server.
func KubeadmCertEtcdHealthcheck() *KubeadmCert {
	return &KubeadmCert{
		Name:     "etcd-healthcheck-client",
		LongName: "certificate for liveness probes to healthcheck etcd",
		BaseName: kubeadmconstants.EtcdHealthcheckClientCertAndKeyBaseName,
		CAName:   "etcd-ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: kubeadmconstants.EtcdHealthcheckClientCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// KubeadmCertEtcdAPIClient is the definition of the cert used by the API server to access etcd.
func KubeadmCertEtcdAPIClient() *KubeadmCert {
	return &KubeadmCert{
		Name:     "apiserver-etcd-client",
		LongName: "certificate the apiserver uses to access etcd",
		BaseName: kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName,
		CAName:   "etcd-ca",
		config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: kubeadmconstants.APIServerEtcdClientCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

func makeAltNamesMutator(f func(*kubeadmapi.InitConfiguration) (*certutil.AltNames, error)) configMutatorsFunc {
	return func(mc *kubeadmapi.InitConfiguration, cc *pkiutil.CertConfig) error {
		altNames, err := f(mc)
		if err != nil {
			return err
		}
		cc.AltNames = *altNames
		return nil
	}
}

func setCommonNameToNodeName() configMutatorsFunc {
	return func(mc *kubeadmapi.InitConfiguration, cc *pkiutil.CertConfig) error {
		cc.CommonName = mc.NodeRegistration.Name
		return nil
	}
}

// leafCertificates returns non-CA certificates from the supplied Certificates.
func leafCertificates(c Certificates) (Certificates, error) {
	certTree, err := c.AsMap().CertTree()
	if err != nil {
		return nil, err
	}

	var out Certificates
	for _, leafCertificates := range certTree {
		out = append(out, leafCertificates...)
	}
	return out, nil
}

func createKeyAndCSR(kubeadmConfig *kubeadmapi.InitConfiguration, cert *KubeadmCert) error {
	if kubeadmConfig == nil {
		return errors.Errorf("%s: kubeadmConfig was nil", errInvalid)
	}
	if cert == nil {
		return errors.Errorf("%s: cert was nil", errInvalid)
	}
	certDir := kubeadmConfig.CertificatesDir
	name := cert.BaseName
	if pkiutil.CSROrKeyExist(certDir, name) {
		return errors.Errorf("%s: key or CSR %s/%s", errExist, certDir, name)
	}
	cfg, err := cert.GetConfig(kubeadmConfig)
	if err != nil {
		return err
	}
	csr, key, err := pkiutil.NewCSRAndKey(cfg)
	if err != nil {
		return err
	}
	err = pkiutil.WriteKey(certDir, name, key)
	if err != nil {
		return err
	}
	return pkiutil.WriteCSR(certDir, name, csr)
}

// CreateDefaultKeysAndCSRFiles is used in ExternalCA mode to create key files
// and adjacent CSR files.
func CreateDefaultKeysAndCSRFiles(out io.Writer, config *kubeadmapi.InitConfiguration) error {
	certificates, err := leafCertificates(GetDefaultCertList())
	if err != nil {
		return err
	}
	if out != nil {
		fmt.Fprintf(out, "generating keys and CSRs in %s\n", config.CertificatesDir)
	}
	for _, cert := range certificates {
		if err := createKeyAndCSR(config, cert); err != nil {
			return err
		}
		if out != nil {
			fmt.Fprintf(out, "  %s\n", cert.BaseName)
		}
	}
	return nil
}
