/*
Copyright 2016 The Kubernetes Authors.

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

package pkiutil

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const (
	// PublicKeyBlockType is a possible value for pem.Block.Type.
	PublicKeyBlockType = "PUBLIC KEY"
	// CertificateBlockType is a possible value for pem.Block.Type.
	CertificateBlockType = "CERTIFICATE"
)

// CertConfig is a wrapper around certutil.Config extending it with EncryptionAlgorithm.
type CertConfig struct {
	certutil.Config
	NotAfter            time.Time
	EncryptionAlgorithm kubeadmapi.EncryptionAlgorithmType
}

// NewCertificateAuthority creates new certificate and private key for the certificate authority
func NewCertificateAuthority(config *CertConfig) (*x509.Certificate, crypto.Signer, error) {
	key, err := NewPrivateKey(config.EncryptionAlgorithm)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create private key while generating CA certificate")
	}
	cert, err := NewSelfSignedCACert(config, key)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create self-signed CA certificate")
	}

	return cert, key, nil
}

// NewIntermediateCertificateAuthority creates new certificate and private key for an intermediate certificate authority
func NewIntermediateCertificateAuthority(parentCert *x509.Certificate, parentKey crypto.Signer, config *CertConfig) (*x509.Certificate, crypto.Signer, error) {
	key, err := NewPrivateKey(config.EncryptionAlgorithm)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create private key while generating intermediate CA certificate")
	}

	cert, err := NewSignedCert(config, key, parentCert, parentKey, true)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to sign intermediate CA certificate")
	}

	return cert, key, nil
}

// NewCertAndKey creates new certificate and key by passing the certificate authority certificate and key
func NewCertAndKey(caCert *x509.Certificate, caKey crypto.Signer, config *CertConfig) (*x509.Certificate, crypto.Signer, error) {
	if len(config.Usages) == 0 {
		return nil, nil, errors.New("must specify at least one ExtKeyUsage")
	}

	key, err := NewPrivateKey(config.EncryptionAlgorithm)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create private key")
	}

	cert, err := NewSignedCert(config, key, caCert, caKey, false)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to sign certificate")
	}

	return cert, key, nil
}

// NewCSRAndKey generates a new key and CSR and that could be signed to create the given certificate
func NewCSRAndKey(config *CertConfig) (*x509.CertificateRequest, crypto.Signer, error) {
	key, err := NewPrivateKey(config.EncryptionAlgorithm)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create private key")
	}

	csr, err := NewCSR(*config, key)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to generate CSR")
	}

	return csr, key, nil
}

// HasServerAuth returns true if the given certificate is a ServerAuth
func HasServerAuth(cert *x509.Certificate) bool {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageServerAuth {
			return true
		}
	}
	return false
}

// WriteCertAndKey stores certificate and key at the specified location
func WriteCertAndKey(pkiPath string, name string, cert *x509.Certificate, key crypto.Signer) error {
	if err := WriteKey(pkiPath, name, key); err != nil {
		return errors.Wrap(err, "couldn't write key")
	}

	return WriteCert(pkiPath, name, cert)
}

// WriteCert stores the given certificate at the given location
func WriteCert(pkiPath, name string, cert *x509.Certificate) error {
	if cert == nil {
		return errors.New("certificate cannot be nil when writing to file")
	}

	certificatePath := pathForCert(pkiPath, name)
	if err := certutil.WriteCert(certificatePath, EncodeCertPEM(cert)); err != nil {
		return errors.Wrapf(err, "unable to write certificate to file %s", certificatePath)
	}

	return nil
}

// WriteCertBundle stores the given certificate bundle at the given location
func WriteCertBundle(pkiPath, name string, certs []*x509.Certificate) error {
	for i, cert := range certs {
		if cert == nil {
			return errors.Errorf("found nil certificate at position %d when writing bundle to file", i)
		}
	}

	certificatePath := pathForCert(pkiPath, name)
	encoded, err := EncodeCertBundlePEM(certs)
	if err != nil {
		return errors.Wrapf(err, "unable to marshal certificate bundle to PEM")
	}
	if err := certutil.WriteCert(certificatePath, encoded); err != nil {
		return errors.Wrapf(err, "unable to write certificate bundle to file %s", certificatePath)
	}

	return nil
}

// WriteKey stores the given key at the given location
func WriteKey(pkiPath, name string, key crypto.Signer) error {
	if key == nil {
		return errors.New("private key cannot be nil when writing to file")
	}

	privateKeyPath := pathForKey(pkiPath, name)
	encoded, err := keyutil.MarshalPrivateKeyToPEM(key)
	if err != nil {
		return errors.Wrapf(err, "unable to marshal private key to PEM")
	}
	if err := keyutil.WriteKey(privateKeyPath, encoded); err != nil {
		return errors.Wrapf(err, "unable to write private key to file %s", privateKeyPath)
	}

	return nil
}

// WriteCSR writes the pem-encoded CSR data to csrPath.
// The CSR file will be created with file mode 0600.
// If the CSR file already exists, it will be overwritten.
// The parent directory of the csrPath will be created as needed with file mode 0700.
func WriteCSR(csrDir, name string, csr *x509.CertificateRequest) error {
	if csr == nil {
		return errors.New("certificate request cannot be nil when writing to file")
	}

	csrPath := pathForCSR(csrDir, name)
	if err := os.MkdirAll(filepath.Dir(csrPath), os.FileMode(0700)); err != nil {
		return errors.Wrapf(err, "failed to make directory %s", filepath.Dir(csrPath))
	}

	if err := os.WriteFile(csrPath, EncodeCSRPEM(csr), os.FileMode(0600)); err != nil {
		return errors.Wrapf(err, "unable to write CSR to file %s", csrPath)
	}

	return nil
}

// WritePublicKey stores the given public key at the given location
func WritePublicKey(pkiPath, name string, key crypto.PublicKey) error {
	if key == nil {
		return errors.New("public key cannot be nil when writing to file")
	}

	publicKeyBytes, err := EncodePublicKeyPEM(key)
	if err != nil {
		return err
	}
	publicKeyPath := pathForPublicKey(pkiPath, name)
	if err := keyutil.WriteKey(publicKeyPath, publicKeyBytes); err != nil {
		return errors.Wrapf(err, "unable to write public key to file %s", publicKeyPath)
	}

	return nil
}

// CertOrKeyExist returns a boolean whether the cert or the key exists
func CertOrKeyExist(pkiPath, name string) bool {
	certificatePath, privateKeyPath := PathsForCertAndKey(pkiPath, name)

	_, certErr := os.Stat(certificatePath)
	_, keyErr := os.Stat(privateKeyPath)

	return !(os.IsNotExist(certErr) && os.IsNotExist(keyErr))
}

// CSROrKeyExist returns true if one of the CSR or key exists
func CSROrKeyExist(csrDir, name string) bool {
	csrPath := pathForCSR(csrDir, name)
	keyPath := pathForKey(csrDir, name)

	_, csrErr := os.Stat(csrPath)
	_, keyErr := os.Stat(keyPath)

	return !(os.IsNotExist(csrErr) && os.IsNotExist(keyErr))
}

// TryLoadCertAndKeyFromDisk tries to load a cert and a key from the disk and validates that they are valid
func TryLoadCertAndKeyFromDisk(pkiPath, name string) (*x509.Certificate, crypto.Signer, error) {
	cert, err := TryLoadCertFromDisk(pkiPath, name)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to load certificate")
	}

	key, err := TryLoadKeyFromDisk(pkiPath, name)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to load key")
	}

	return cert, key, nil
}

// TryLoadCertFromDisk tries to load the cert from the disk
func TryLoadCertFromDisk(pkiPath, name string) (*x509.Certificate, error) {
	certificatePath := pathForCert(pkiPath, name)

	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't load the certificate file %s", certificatePath)
	}

	// Safely pick the first one because the sender's certificate must come first in the list.
	// For details, see: https://www.rfc-editor.org/rfc/rfc4346#section-7.4.2
	cert := certs[0]

	return cert, nil
}

// TryLoadCertChainFromDisk tries to load the cert chain from the disk
func TryLoadCertChainFromDisk(pkiPath, name string) (*x509.Certificate, []*x509.Certificate, error) {
	certificatePath := pathForCert(pkiPath, name)

	certs, err := certutil.CertsFromFile(certificatePath)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't load the certificate file %s", certificatePath)
	}

	cert := certs[0]
	intermediates := certs[1:]

	return cert, intermediates, nil
}

// TryLoadKeyFromDisk tries to load the key from the disk and validates that it is valid
func TryLoadKeyFromDisk(pkiPath, name string) (crypto.Signer, error) {
	privateKeyPath := pathForKey(pkiPath, name)

	// Parse the private key from a file
	privKey, err := keyutil.PrivateKeyFromFile(privateKeyPath)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't load the private key file %s", privateKeyPath)
	}

	// Allow RSA and ECDSA formats only
	var key crypto.Signer
	switch k := privKey.(type) {
	case *rsa.PrivateKey:
		key = k
	case *ecdsa.PrivateKey:
		key = k
	default:
		return nil, errors.Errorf("the private key file %s is neither in RSA nor ECDSA format", privateKeyPath)
	}

	return key, nil
}

// TryLoadPrivatePublicKeyFromDisk tries to load the key from the disk and validates that it is valid
func TryLoadPrivatePublicKeyFromDisk(pkiPath, name string) (crypto.PrivateKey, crypto.PublicKey, error) {
	privateKeyPath := pathForKey(pkiPath, name)

	// Parse the private key from a file
	privKey, err := keyutil.PrivateKeyFromFile(privateKeyPath)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't load the private key file %s", privateKeyPath)
	}

	publicKeyPath := pathForPublicKey(pkiPath, name)

	// Parse the public key from a file
	pubKeys, err := keyutil.PublicKeysFromFile(publicKeyPath)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "couldn't load the public key file %s", publicKeyPath)
	}

	// Allow RSA and ECDSA formats only
	switch k := privKey.(type) {
	case *rsa.PrivateKey:
		return k, pubKeys[0].(*rsa.PublicKey), nil
	case *ecdsa.PrivateKey:
		return k, pubKeys[0].(*ecdsa.PublicKey), nil
	default:
		return nil, nil, errors.Errorf("the private key file %s is neither in RSA nor ECDSA format", privateKeyPath)
	}
}

// PathsForCertAndKey returns the paths for the certificate and key given the path and basename.
func PathsForCertAndKey(pkiPath, name string) (string, string) {
	return pathForCert(pkiPath, name), pathForKey(pkiPath, name)
}

func pathForCert(pkiPath, name string) string {
	return filepath.Join(pkiPath, fmt.Sprintf("%s.crt", name))
}

func pathForKey(pkiPath, name string) string {
	return filepath.Join(pkiPath, fmt.Sprintf("%s.key", name))
}

func pathForPublicKey(pkiPath, name string) string {
	return filepath.Join(pkiPath, fmt.Sprintf("%s.pub", name))
}

func pathForCSR(pkiPath, name string) string {
	return filepath.Join(pkiPath, fmt.Sprintf("%s.csr", name))
}

// GetAPIServerAltNames builds an AltNames object for to be used when generating apiserver certificate
func GetAPIServerAltNames(cfg *kubeadmapi.InitConfiguration) (*certutil.AltNames, error) {
	// advertise address
	advertiseAddress := netutils.ParseIPSloppy(cfg.LocalAPIEndpoint.AdvertiseAddress)
	if advertiseAddress == nil {
		return nil, errors.Errorf("error parsing LocalAPIEndpoint AdvertiseAddress %v: is not a valid textual representation of an IP address",
			cfg.LocalAPIEndpoint.AdvertiseAddress)
	}

	internalAPIServerVirtualIP, err := kubeadmconstants.GetAPIServerVirtualIP(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to get first IP address from the given CIDR: %v", cfg.Networking.ServiceSubnet)
	}

	// create AltNames with defaults DNSNames/IPs
	altNames := &certutil.AltNames{
		DNSNames: []string{
			cfg.NodeRegistration.Name,
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			fmt.Sprintf("kubernetes.default.svc.%s", cfg.Networking.DNSDomain),
		},
		IPs: []net.IP{
			internalAPIServerVirtualIP,
			advertiseAddress,
		},
	}

	// add cluster controlPlaneEndpoint if present (dns or ip)
	if len(cfg.ControlPlaneEndpoint) > 0 {
		if host, _, err := kubeadmutil.ParseHostPort(cfg.ControlPlaneEndpoint); err == nil {
			if ip := netutils.ParseIPSloppy(host); ip != nil {
				altNames.IPs = append(altNames.IPs, ip)
			} else {
				altNames.DNSNames = append(altNames.DNSNames, host)
			}
		} else {
			return nil, errors.Wrapf(err, "error parsing cluster controlPlaneEndpoint %q", cfg.ControlPlaneEndpoint)
		}
	}

	appendSANsToAltNames(altNames, cfg.APIServer.CertSANs, kubeadmconstants.APIServerCertName)

	return altNames, nil
}

// GetEtcdAltNames builds an AltNames object for generating the etcd server certificate.
// `advertise address` and localhost are included in the SAN since this is the interfaces the etcd static pod listens on.
// The user can override the listen address with `Etcd.ExtraArgs` and add SANs with `Etcd.ServerCertSANs`.
func GetEtcdAltNames(cfg *kubeadmapi.InitConfiguration) (*certutil.AltNames, error) {
	return getAltNames(cfg, kubeadmconstants.EtcdServerCertName)
}

// GetEtcdPeerAltNames builds an AltNames object for generating the etcd peer certificate.
// Hostname and `API.AdvertiseAddress` are included if the user chooses to promote the single node etcd cluster into a multi-node one (stacked etcd).
// The user can override the listen address with `Etcd.ExtraArgs` and add SANs with `Etcd.PeerCertSANs`.
func GetEtcdPeerAltNames(cfg *kubeadmapi.InitConfiguration) (*certutil.AltNames, error) {
	return getAltNames(cfg, kubeadmconstants.EtcdPeerCertName)
}

// getAltNames builds an AltNames object with the cfg and certName.
func getAltNames(cfg *kubeadmapi.InitConfiguration, certName string) (*certutil.AltNames, error) {
	// advertise address
	advertiseAddress := netutils.ParseIPSloppy(cfg.LocalAPIEndpoint.AdvertiseAddress)
	if advertiseAddress == nil {
		return nil, errors.Errorf("error parsing LocalAPIEndpoint AdvertiseAddress %v: is not a valid textual representation of an IP address",
			cfg.LocalAPIEndpoint.AdvertiseAddress)
	}

	// create AltNames with defaults DNSNames/IPs
	altNames := &certutil.AltNames{
		DNSNames: []string{cfg.NodeRegistration.Name, "localhost"},
		IPs:      []net.IP{advertiseAddress, net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}

	if cfg.Etcd.Local != nil {
		if certName == kubeadmconstants.EtcdServerCertName {
			appendSANsToAltNames(altNames, cfg.Etcd.Local.ServerCertSANs, kubeadmconstants.EtcdServerCertName)
		} else if certName == kubeadmconstants.EtcdPeerCertName {
			appendSANsToAltNames(altNames, cfg.Etcd.Local.PeerCertSANs, kubeadmconstants.EtcdPeerCertName)
		}
	}
	return altNames, nil
}

// appendSANsToAltNames parses SANs from as list of strings and adds them to altNames for use on a specific cert
// altNames is passed in with a pointer, and the struct is modified
// valid IP address strings are parsed and added to altNames.IPs as net.IP's
// RFC-1123 compliant DNS strings are added to altNames.DNSNames as strings
// RFC-1123 compliant wildcard DNS strings are added to altNames.DNSNames as strings
// certNames is used to print user facing warnings and should be the name of the cert the altNames will be used for
func appendSANsToAltNames(altNames *certutil.AltNames, SANs []string, certName string) {
	for _, altname := range SANs {
		if ip := netutils.ParseIPSloppy(altname); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		} else if len(validation.IsWildcardDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		} else {
			klog.Warningf(
				"[certificates] WARNING: '%s' was not added to the '%s' SAN, because it is not a valid IP or RFC-1123 compliant DNS entry\n",
				altname,
				certName,
			)
		}
	}
}

// EncodeCSRPEM returns PEM-encoded CSR data
func EncodeCSRPEM(csr *x509.CertificateRequest) []byte {
	block := pem.Block{
		Type:  certutil.CertificateRequestBlockType,
		Bytes: csr.Raw,
	}
	return pem.EncodeToMemory(&block)
}

// NewCSR creates a new CSR
func NewCSR(cfg CertConfig, key crypto.Signer) (*x509.CertificateRequest, error) {
	RemoveDuplicateAltNames(&cfg.AltNames)

	template := &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: cfg.Organization,
		},
		DNSNames:    cfg.AltNames.DNSNames,
		IPAddresses: cfg.AltNames.IPs,
	}

	csrBytes, err := x509.CreateCertificateRequest(cryptorand.Reader, template, key)

	if err != nil {
		return nil, errors.Wrap(err, "failed to create a CSR")
	}

	return x509.ParseCertificateRequest(csrBytes)
}

// EncodeCertPEM returns PEM-encoded certificate data
func EncodeCertPEM(cert *x509.Certificate) []byte {
	block := pem.Block{
		Type:  CertificateBlockType,
		Bytes: cert.Raw,
	}
	return pem.EncodeToMemory(&block)
}

// EncodeCertBundlePEM returns PEM-encoded certificate bundle
func EncodeCertBundlePEM(certs []*x509.Certificate) ([]byte, error) {
	buf := bytes.Buffer{}

	block := pem.Block{
		Type: CertificateBlockType,
	}

	for _, cert := range certs {
		block.Bytes = cert.Raw
		if err := pem.Encode(&buf, &block); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// EncodePublicKeyPEM returns PEM-encoded public data
func EncodePublicKeyPEM(key crypto.PublicKey) ([]byte, error) {
	der, err := x509.MarshalPKIXPublicKey(key)
	if err != nil {
		return []byte{}, err
	}
	block := pem.Block{
		Type:  PublicKeyBlockType,
		Bytes: der,
	}
	return pem.EncodeToMemory(&block), nil
}

// NewPrivateKey returns a new private key.
var NewPrivateKey = GeneratePrivateKey

// rsaKeySizeFromAlgorithmType takes a known RSA algorithm defined in the kubeadm API
// an returns its key size. For unknown types it returns 0. For an empty type it returns
// the default size of 2048.
func rsaKeySizeFromAlgorithmType(keyType kubeadmapi.EncryptionAlgorithmType) int {
	switch keyType {
	case kubeadmapi.EncryptionAlgorithmRSA2048, "":
		return 2048
	case kubeadmapi.EncryptionAlgorithmRSA3072:
		return 3072
	case kubeadmapi.EncryptionAlgorithmRSA4096:
		return 4096
	default:
		return 0
	}
}

// GeneratePrivateKey is the default function for generating private keys.
func GeneratePrivateKey(keyType kubeadmapi.EncryptionAlgorithmType) (crypto.Signer, error) {
	if keyType == kubeadmapi.EncryptionAlgorithmECDSAP256 {
		return ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	}

	rsaKeySize := rsaKeySizeFromAlgorithmType(keyType)
	if rsaKeySize == 0 {
		return nil, errors.Errorf("cannot obtain key size from unknown RSA algorithm: %q", keyType)
	}
	return rsa.GenerateKey(cryptorand.Reader, rsaKeySize)
}

// NewSignedCert creates a signed certificate using the given CA certificate and key
func NewSignedCert(cfg *CertConfig, key crypto.Signer, caCert *x509.Certificate, caKey crypto.Signer, isCA bool) (*x509.Certificate, error) {
	// returns a uniform random value in [0, max-1), then add 1 to serial to make it a uniform random value in [1, max).
	serial, err := cryptorand.Int(cryptorand.Reader, new(big.Int).SetInt64(math.MaxInt64-1))
	if err != nil {
		return nil, err
	}
	serial = new(big.Int).Add(serial, big.NewInt(1))
	if len(cfg.CommonName) == 0 {
		return nil, errors.New("must specify a CommonName")
	}

	keyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature
	if isCA {
		keyUsage |= x509.KeyUsageCertSign
	}

	RemoveDuplicateAltNames(&cfg.AltNames)

	notBefore := caCert.NotBefore
	if !cfg.NotBefore.IsZero() {
		notBefore = cfg.NotBefore
	}

	notAfter := notBefore.Add(kubeadmconstants.CertificateValidityPeriod)
	if !cfg.NotAfter.IsZero() {
		notAfter = cfg.NotAfter
	}

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: cfg.Organization,
		},
		DNSNames:              cfg.AltNames.DNSNames,
		IPAddresses:           cfg.AltNames.IPs,
		SerialNumber:          serial,
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              keyUsage,
		ExtKeyUsage:           cfg.Usages,
		BasicConstraintsValid: true,
		IsCA:                  isCA,
	}
	certDERBytes, err := x509.CreateCertificate(cryptorand.Reader, &certTmpl, caCert, key.Public(), caKey)
	if err != nil {
		return nil, err
	}
	return x509.ParseCertificate(certDERBytes)
}

// NewSelfSignedCACert creates a new self-signed CA certificate
func NewSelfSignedCACert(cfg *CertConfig, key crypto.Signer) (*x509.Certificate, error) {
	// returns a uniform random value in [0, max-1), then add 1 to serial to make it a uniform random value in [1, max).
	serial, err := cryptorand.Int(cryptorand.Reader, new(big.Int).SetInt64(math.MaxInt64-1))
	if err != nil {
		return nil, err
	}
	serial = new(big.Int).Add(serial, big.NewInt(1))

	keyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign

	notBefore := time.Now().UTC()
	if !cfg.NotBefore.IsZero() {
		notBefore = cfg.NotBefore
	}

	notAfter := notBefore.Add(kubeadmconstants.CACertificateValidityPeriod)
	if !cfg.NotAfter.IsZero() {
		notAfter = cfg.NotAfter
	}

	tmpl := x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: cfg.Organization,
		},
		DNSNames:              []string{cfg.CommonName},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              keyUsage,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	certDERBytes, err := x509.CreateCertificate(cryptorand.Reader, &tmpl, &tmpl, key.Public(), key)
	if err != nil {
		return nil, err
	}
	return x509.ParseCertificate(certDERBytes)
}

// RemoveDuplicateAltNames removes duplicate items in altNames.
func RemoveDuplicateAltNames(altNames *certutil.AltNames) {
	if altNames == nil {
		return
	}

	if altNames.DNSNames != nil {
		altNames.DNSNames = sets.List(sets.New(altNames.DNSNames...))
	}

	ipsKeys := make(map[string]struct{})
	var ips []net.IP
	for _, one := range altNames.IPs {
		if _, ok := ipsKeys[one.String()]; !ok {
			ipsKeys[one.String()] = struct{}{}
			ips = append(ips, one)
		}
	}
	altNames.IPs = ips
}

// ValidateCertPeriod checks if the certificate is valid relative to the current time
// (+/- offset)
func ValidateCertPeriod(cert *x509.Certificate, offset time.Duration) error {
	period := fmt.Sprintf("NotBefore: %v, NotAfter: %v", cert.NotBefore, cert.NotAfter)
	now := time.Now().Add(offset).UTC()
	if now.Before(cert.NotBefore) {
		return errors.Errorf("the certificate is not valid yet: %s", period)
	}
	if now.After(cert.NotAfter) {
		return errors.Errorf("the certificate has expired: %s", period)
	}
	return nil
}

// VerifyCertChain verifies that a certificate has a valid chain of
// intermediate CAs back to the root CA
func VerifyCertChain(cert *x509.Certificate, intermediates []*x509.Certificate, root *x509.Certificate) error {
	rootPool := x509.NewCertPool()
	rootPool.AddCert(root)

	intermediatePool := x509.NewCertPool()
	for _, c := range intermediates {
		intermediatePool.AddCert(c)
	}

	verifyOptions := x509.VerifyOptions{
		Roots:         rootPool,
		Intermediates: intermediatePool,
		KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageAny},
	}

	if _, err := cert.Verify(verifyOptions); err != nil {
		return err
	}

	return nil
}
