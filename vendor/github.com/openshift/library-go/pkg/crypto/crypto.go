package crypto

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	mathrand "math/rand"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/util/cert"

	configv1 "github.com/openshift/api/config/v1"
)

// TLS configuration
//
// The OpenShift API defines TLS profiles using OpenSSL cipher suite names.
// Go's crypto/tls uses IANA-standard cipher suite names (which happen to
// match the Go constant names). This package bridges the two naming schemes:
// OpenSSLToIANACipherSuites translates API-level OpenSSL names into the IANA
// names that Go understands, silently dropping any cipher that Go's crypto/tls
// cannot negotiate. This silent narrowing is by design: Go literally cannot
// serve those ciphers, so listing them would be misleading.

// goTLSVersions lists all TLS versions that Go's crypto/tls can negotiate.
// Kept in sync with the crypto/tls package by TestConstantMaps.
var goTLSVersions = map[string]uint16{
	"VersionTLS10": tls.VersionTLS10,
	"VersionTLS11": tls.VersionTLS11,
	"VersionTLS12": tls.VersionTLS12,
	"VersionTLS13": tls.VersionTLS13,
}

// enabledTLSVersions is the subset of goTLSVersions that OpenShift allows
// in TLS configurations. Remove an entry here (not from goTLSVersions) to
// phase out a version while still being able to parse legacy references.
var enabledTLSVersions = map[string]uint16{
	"VersionTLS10": tls.VersionTLS10,
	"VersionTLS11": tls.VersionTLS11,
	"VersionTLS12": tls.VersionTLS12,
	"VersionTLS13": tls.VersionTLS13,
}

// TLSVersionToNameOrDie given a tls version as an int, return its readable name
func TLSVersionToNameOrDie(intVal uint16) string {
	matches := []string{}
	for key, version := range goTLSVersions {
		if version == intVal {
			matches = append(matches, key)
		}
	}

	if len(matches) == 0 {
		panic(fmt.Sprintf("no name found for %d", intVal))
	}
	if len(matches) > 1 {
		panic(fmt.Sprintf("multiple names found for %d: %v", intVal, matches))
	}
	return matches[0]
}

func TLSVersion(versionName string) (uint16, error) {
	if len(versionName) == 0 {
		return DefaultTLSVersion(), nil
	}
	if version, ok := goTLSVersions[versionName]; ok {
		return version, nil
	}
	return 0, fmt.Errorf("unknown tls version %q", versionName)
}
func TLSVersionOrDie(versionName string) uint16 {
	version, err := TLSVersion(versionName)
	if err != nil {
		panic(err)
	}
	return version
}

// GolangTLSVersions returns all TLS versions known to this Go build.
//
// Deprecated: Use ValidTLSVersions instead.
func GolangTLSVersions() []string {
	supported := []string{}
	for k := range goTLSVersions {
		supported = append(supported, k)
	}
	sort.Strings(supported)
	return supported
}

// ValidTLSVersions returns the TLS versions that OpenShift allows in configurations.
func ValidTLSVersions() []string {
	validVersions := []string{}
	for k := range enabledTLSVersions {
		validVersions = append(validVersions, k)
	}
	sort.Strings(validVersions)
	return validVersions
}
func DefaultTLSVersion() uint16 {
	// Can't use SSLv3 because of POODLE and BEAST
	// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
	// Can't use TLSv1.1 because of RC4 cipher usage
	return tls.VersionTLS12
}

// goCipherSuites lists all cipher suites recognized by Go's crypto/tls, keyed
// by IANA name. Kept in sync with the crypto/tls package by TestConstantMaps.
var goCipherSuites = map[string]uint16{
	"TLS_RSA_WITH_RC4_128_SHA":                      tls.TLS_RSA_WITH_RC4_128_SHA,
	"TLS_RSA_WITH_3DES_EDE_CBC_SHA":                 tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA,
	"TLS_RSA_WITH_AES_128_CBC_SHA":                  tls.TLS_RSA_WITH_AES_128_CBC_SHA,
	"TLS_RSA_WITH_AES_256_CBC_SHA":                  tls.TLS_RSA_WITH_AES_256_CBC_SHA,
	"TLS_RSA_WITH_AES_128_CBC_SHA256":               tls.TLS_RSA_WITH_AES_128_CBC_SHA256,
	"TLS_RSA_WITH_AES_128_GCM_SHA256":               tls.TLS_RSA_WITH_AES_128_GCM_SHA256,
	"TLS_RSA_WITH_AES_256_GCM_SHA384":               tls.TLS_RSA_WITH_AES_256_GCM_SHA384,
	"TLS_ECDHE_ECDSA_WITH_RC4_128_SHA":              tls.TLS_ECDHE_ECDSA_WITH_RC4_128_SHA,
	"TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA":          tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
	"TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA":          tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
	"TLS_ECDHE_RSA_WITH_RC4_128_SHA":                tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA,
	"TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA":           tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA,
	"TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA":            tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
	"TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA":            tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
	"TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256":       tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,
	"TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256":         tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256,
	"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256":         tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
	"TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256":       tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
	"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384":         tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
	"TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384":       tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
	"TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305":          tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
	"TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305":        tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
	"TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256":   tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
	"TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256": tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
	"TLS_AES_128_GCM_SHA256":                        tls.TLS_AES_128_GCM_SHA256,
	"TLS_AES_256_GCM_SHA384":                        tls.TLS_AES_256_GCM_SHA384,
	"TLS_CHACHA20_POLY1305_SHA256":                  tls.TLS_CHACHA20_POLY1305_SHA256,
}

// openSSLToIANACiphers maps OpenSSL cipher suite names to IANA names for
// every cipher that Go's crypto/tls can negotiate.
// Ref: https://www.iana.org/assignments/tls-parameters/tls-parameters.xml
// Ciphers defined in the API but absent from Go are tracked in
// ciphersUnsupportedByGo (below) so tests detect when Go gains support.
var openSSLToIANACiphers = map[string]string{
	// TLS 1.3 ciphers - always negotiated by Go; not individually configurable.
	"TLS_AES_128_GCM_SHA256":       "TLS_AES_128_GCM_SHA256",       // 0x13,0x01
	"TLS_AES_256_GCM_SHA384":       "TLS_AES_256_GCM_SHA384",       // 0x13,0x02
	"TLS_CHACHA20_POLY1305_SHA256": "TLS_CHACHA20_POLY1305_SHA256", // 0x13,0x03

	// TLS 1.2
	"ECDHE-ECDSA-AES128-GCM-SHA256": "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",       // 0xC0,0x2B
	"ECDHE-RSA-AES128-GCM-SHA256":   "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",         // 0xC0,0x2F
	"ECDHE-ECDSA-AES256-GCM-SHA384": "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",       // 0xC0,0x2C
	"ECDHE-RSA-AES256-GCM-SHA384":   "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",         // 0xC0,0x30
	"ECDHE-ECDSA-CHACHA20-POLY1305": "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", // 0xCC,0xA9
	"ECDHE-RSA-CHACHA20-POLY1305":   "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",   // 0xCC,0xA8
	"ECDHE-ECDSA-AES128-SHA256":     "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256",       // 0xC0,0x23
	"ECDHE-RSA-AES128-SHA256":       "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",         // 0xC0,0x27
	"AES128-GCM-SHA256":             "TLS_RSA_WITH_AES_128_GCM_SHA256",               // 0x00,0x9C
	"AES256-GCM-SHA384":             "TLS_RSA_WITH_AES_256_GCM_SHA384",               // 0x00,0x9D
	"AES128-SHA256":                 "TLS_RSA_WITH_AES_128_CBC_SHA256",               // 0x00,0x3C

	// Ciphers defined in the API but not supported by Go are listed in
	// ciphersUnsupportedByGo below.

	// TLS 1
	"ECDHE-ECDSA-AES128-SHA": "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", // 0xC0,0x09
	"ECDHE-RSA-AES128-SHA":   "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",   // 0xC0,0x13
	"ECDHE-ECDSA-AES256-SHA": "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", // 0xC0,0x0A
	"ECDHE-RSA-AES256-SHA":   "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",   // 0xC0,0x14

	// SSL 3
	"AES128-SHA":             "TLS_RSA_WITH_AES_128_CBC_SHA",        // 0x00,0x2F
	"AES256-SHA":             "TLS_RSA_WITH_AES_256_CBC_SHA",        // 0x00,0x35
	"DES-CBC3-SHA":           "TLS_RSA_WITH_3DES_EDE_CBC_SHA",       // 0x00,0x0A
	"ECDHE-RSA-DES-CBC3-SHA": "TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA", // 0xC0,0x12
}

// ciphersUnsupportedByGo lists cipher suites that are defined in the OpenShift API
// TLS profiles (from the Mozilla guidelines) but are not supported by Go's crypto/tls.
// These are intentionally excluded from openSSLToIANACiphers and silently filtered
// out during profile translation. The IANA names come from the IANA TLS Cipher Suite
// Registry (https://www.iana.org/assignments/tls-parameters/) and are retained so
// TestCiphersUnsupportedByGoAreActuallyUnsupported can detect if a future Go
// release adds support.
var ciphersUnsupportedByGo = map[string]string{
	"ECDHE-ECDSA-AES256-SHA384": "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384",
	"ECDHE-RSA-AES256-SHA384":   "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
	"AES256-SHA256":             "TLS_RSA_WITH_AES_256_CBC_SHA256",
}

// CipherSuitesToNamesOrDie given a list of cipher suites as ints, return their readable names
func CipherSuitesToNamesOrDie(intVals []uint16) []string {
	ret := []string{}
	for _, intVal := range intVals {
		ret = append(ret, CipherSuiteToNameOrDie(intVal))
	}

	return ret
}

// CipherSuiteToNameOrDie given a cipher suite as an int, return its readable name
func CipherSuiteToNameOrDie(intVal uint16) string {
	// The following suite ids appear twice in the cipher map (with
	// and without the _SHA256 suffix) for the purposes of backwards
	// compatibility. Always return the current rather than the legacy
	// name.
	switch intVal {
	case tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:
		return "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"
	case tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:
		return "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256"
	}

	matches := []string{}
	for key, version := range goCipherSuites {
		if version == intVal {
			matches = append(matches, key)
		}
	}

	if len(matches) == 0 {
		panic(fmt.Sprintf("no name found for %d", intVal))
	}
	if len(matches) > 1 {
		panic(fmt.Sprintf("multiple names found for %d: %v", intVal, matches))
	}
	return matches[0]
}

func CipherSuite(cipherName string) (uint16, error) {
	if cipher, ok := goCipherSuites[cipherName]; ok {
		return cipher, nil
	}

	return 0, fmt.Errorf("unknown cipher name %q", cipherName)
}

func CipherSuitesOrDie(cipherNames []string) []uint16 {
	if len(cipherNames) == 0 {
		return DefaultCiphers()
	}
	cipherValues := []uint16{}
	for _, cipherName := range cipherNames {
		cipher, err := CipherSuite(cipherName)
		if err != nil {
			panic(err)
		}
		cipherValues = append(cipherValues, cipher)
	}
	return cipherValues
}
func ValidCipherSuites() []string {
	validCipherSuites := []string{}
	for k := range goCipherSuites {
		validCipherSuites = append(validCipherSuites, k)
	}
	sort.Strings(validCipherSuites)
	return validCipherSuites
}

// DefaultTLSProfileType is the intermediate profile type.
const DefaultTLSProfileType = configv1.TLSProfileIntermediateType

// DefaultCiphers returns the default cipher suites for TLS connections.
//
// RECOMMENDATION: Instead of relying on this function directly, consumers should respect
// TLSSecurityProfile settings from one of the OpenShift API configuration resources:
//   - For API servers: Use apiserver.config.openshift.io/cluster Spec.TLSSecurityProfile
//   - For ingress controllers: Use operator.openshift.io/v1 IngressController Spec.TLSSecurityProfile
//   - For kubelet: Use machineconfiguration.openshift.io/v1 KubeletConfig Spec.TLSSecurityProfile
//
// These API resources allow cluster administrators to choose between Old, Intermediate,
// Modern, or Custom TLS profiles. Components should observe these settings.
func DefaultCiphers() []uint16 {
	// Aligned with intermediate profile of the 5.7 version of the Mozilla Server
	// Side TLS guidelines found at: https://ssl-config.mozilla.org/guidelines/5.7.json
	//
	// Latest guidelines: https://ssl-config.mozilla.org/guidelines/latest.json
	//
	// This profile provides strong security with wide compatibility.
	// It requires TLS 1.2+ and uses only AEAD cipher suites (GCM, ChaCha20-Poly1305)
	// with ECDHE key exchange for perfect forward secrecy.
	//
	// All CBC-mode ciphers have been removed due to padding oracle vulnerabilities.
	// All RSA key exchange ciphers have been removed due to lack of perfect forward secrecy.
	//
	// HTTP/2 compliance: All ciphers are compliant with RFC7540, section 9.2.
	return []uint16{
		// TLS 1.2 cipher suites with ECDHE + AEAD
		tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
		tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, // required by HTTP/2
		tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,

		// TLS 1.3 cipher suites (negotiated automatically, not configurable)
		tls.TLS_AES_128_GCM_SHA256,
		tls.TLS_AES_256_GCM_SHA384,
		tls.TLS_CHACHA20_POLY1305_SHA256,
	}
}

// SecureTLSConfig enforces the default minimum security settings for the cluster.
func SecureTLSConfig(config *tls.Config) *tls.Config {
	if config.MinVersion == 0 {
		config.MinVersion = DefaultTLSVersion()
	}

	config.PreferServerCipherSuites = true
	if len(config.CipherSuites) == 0 {
		config.CipherSuites = DefaultCiphers()
	}
	return config
}

// OpenSSLToIANACipherSuites maps input OpenSSL Cipher Suite names to their
// IANA counterparts.
// Ciphers that Go's crypto/tls cannot negotiate are silently dropped and
// logged at V(4).
func OpenSSLToIANACipherSuites(ciphers []string) []string {
	ianaCiphers := make([]string, 0, len(ciphers))

	for _, c := range ciphers {
		ianaCipher, found := openSSLToIANACiphers[c]
		if found {
			ianaCiphers = append(ianaCiphers, ianaCipher)
		} else {
			klog.V(4).Infof("Dropping cipher %q: not supported by Go's crypto/tls", c)
		}
	}

	return ianaCiphers
}

type TLSCertificateConfig struct {
	Certs []*x509.Certificate
	Key   crypto.PrivateKey
}

type TLSCARoots struct {
	Roots []*x509.Certificate
}

func (c *TLSCertificateConfig) WriteCertConfigFile(certFile, keyFile string) error {
	// ensure parent dir
	if err := os.MkdirAll(filepath.Dir(certFile), os.FileMode(0755)); err != nil {
		return err
	}
	certFileWriter, err := os.OpenFile(certFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(keyFile), os.FileMode(0755)); err != nil {
		return err
	}
	keyFileWriter, err := os.OpenFile(keyFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}

	if err := writeCertificates(certFileWriter, c.Certs...); err != nil {
		return err
	}
	if err := writeKeyFile(keyFileWriter, c.Key); err != nil {
		return err
	}

	if err := certFileWriter.Close(); err != nil {
		return err
	}
	if err := keyFileWriter.Close(); err != nil {
		return err
	}

	return nil
}

func (c *TLSCertificateConfig) WriteCertConfig(certFile, keyFile io.Writer) error {
	if err := writeCertificates(certFile, c.Certs...); err != nil {
		return err
	}
	if err := writeKeyFile(keyFile, c.Key); err != nil {
		return err
	}
	return nil
}

func (c *TLSCertificateConfig) GetPEMBytes() ([]byte, []byte, error) {
	certBytes, err := EncodeCertificates(c.Certs...)
	if err != nil {
		return nil, nil, err
	}
	keyBytes, err := EncodeKey(c.Key)
	if err != nil {
		return nil, nil, err
	}

	return certBytes, keyBytes, nil
}

func GetTLSCertificateConfig(certFile, keyFile string) (*TLSCertificateConfig, error) {
	if len(certFile) == 0 {
		return nil, errors.New("certFile missing")
	}
	if len(keyFile) == 0 {
		return nil, errors.New("keyFile missing")
	}

	certPEMBlock, err := os.ReadFile(certFile)
	if err != nil {
		return nil, err
	}
	certs, err := cert.ParseCertsPEM(certPEMBlock)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %s", certFile, err)
	}

	keyPEMBlock, err := os.ReadFile(keyFile)
	if err != nil {
		return nil, err
	}
	keyPairCert, err := tls.X509KeyPair(certPEMBlock, keyPEMBlock)
	if err != nil {
		return nil, err
	}
	key := keyPairCert.PrivateKey

	return &TLSCertificateConfig{certs, key}, nil
}

func GetTLSCertificateConfigFromBytes(certBytes, keyBytes []byte) (*TLSCertificateConfig, error) {
	if len(certBytes) == 0 {
		return nil, errors.New("certFile missing")
	}
	if len(keyBytes) == 0 {
		return nil, errors.New("keyFile missing")
	}

	certs, err := cert.ParseCertsPEM(certBytes)
	if err != nil {
		return nil, fmt.Errorf("error reading cert: %s", err)
	}

	keyPairCert, err := tls.X509KeyPair(certBytes, keyBytes)
	if err != nil {
		return nil, err
	}
	key := keyPairCert.PrivateKey

	return &TLSCertificateConfig{certs, key}, nil
}

const (
	DefaultCertificateLifetimeDuration   = time.Hour * 24 * 365 * 2 // 2 years
	DefaultCACertificateLifetimeDuration = time.Hour * 24 * 365 * 5 // 5 years

	// Default keys are 2048 bits
	keyBits = 2048
)

type CA struct {
	Config *TLSCertificateConfig

	SerialGenerator SerialGenerator
}

// SerialGenerator is an interface for getting a serial number for the cert.  It MUST be thread-safe.
type SerialGenerator interface {
	Next(template *x509.Certificate) (int64, error)
}

// SerialFileGenerator returns a unique, monotonically increasing serial number and ensures the CA on disk records that value.
type SerialFileGenerator struct {
	SerialFile string

	// lock guards access to the Serial field
	lock   sync.Mutex
	Serial int64
}

func NewSerialFileGenerator(serialFile string) (*SerialFileGenerator, error) {
	// read serial file, it must already exist
	serial, err := fileToSerial(serialFile)
	if err != nil {
		return nil, err
	}

	generator := &SerialFileGenerator{
		Serial:     serial,
		SerialFile: serialFile,
	}

	// 0 is unused and 1 is reserved for the CA itself
	// Thus we need to guarantee that the first external call to SerialFileGenerator.Next returns 2+
	// meaning that SerialFileGenerator.Serial must not be less than 1 (it is guaranteed to be non-negative)
	if generator.Serial < 1 {
		// fake a call to Next so the file stays in sync and Serial is incremented
		if _, err := generator.Next(&x509.Certificate{}); err != nil {
			return nil, err
		}
	}

	return generator, nil
}

// Next returns a unique, monotonically increasing serial number and ensures the CA on disk records that value.
func (s *SerialFileGenerator) Next(template *x509.Certificate) (int64, error) {
	s.lock.Lock()
	defer s.lock.Unlock()

	// do a best effort check to make sure concurrent external writes are not occurring to the underlying serial file
	serial, err := fileToSerial(s.SerialFile)
	if err != nil {
		return 0, err
	}
	if serial != s.Serial {
		return 0, fmt.Errorf("serial file %s out of sync ram=%d disk=%d", s.SerialFile, s.Serial, serial)
	}

	next := s.Serial + 1
	s.Serial = next

	// Output in hex, padded to multiples of two characters for OpenSSL's sake
	serialText := fmt.Sprintf("%X", next)
	if len(serialText)%2 == 1 {
		serialText = "0" + serialText
	}
	// always add a newline at the end to have a valid file
	serialText += "\n"

	if err := os.WriteFile(s.SerialFile, []byte(serialText), os.FileMode(0640)); err != nil {
		return 0, err
	}
	return next, nil
}

func fileToSerial(serialFile string) (int64, error) {
	serialData, err := os.ReadFile(serialFile)
	if err != nil {
		return 0, err
	}

	// read the file as a single hex number after stripping any whitespace
	serial, err := strconv.ParseInt(string(bytes.TrimSpace(serialData)), 16, 64)
	if err != nil {
		return 0, err
	}

	if serial < 0 {
		return 0, fmt.Errorf("invalid negative serial %d in serial file %s", serial, serialFile)
	}

	return serial, nil
}

// RandomSerialGenerator returns a serial based on time.Now and the subject
type RandomSerialGenerator struct {
}

func (s *RandomSerialGenerator) Next(template *x509.Certificate) (int64, error) {
	return randomSerialNumber(), nil
}

// randomSerialNumber returns a random int64 serial number based on
// time.Now. It is defined separately from the generator interface so
// that the caller doesn't have to worry about an input template or
// error - these are unnecessary when creating a random serial.
func randomSerialNumber() int64 {
	r := mathrand.New(mathrand.NewSource(time.Now().UTC().UnixNano()))
	return r.Int63()
}

// EnsureCA returns a CA, whether it was created (as opposed to pre-existing), and any error
// if serialFile is empty, a RandomSerialGenerator will be used
func EnsureCA(certFile, keyFile, serialFile, name string, lifetime time.Duration) (*CA, bool, error) {
	if ca, err := GetCA(certFile, keyFile, serialFile); err == nil {
		return ca, false, err
	}
	ca, err := MakeSelfSignedCA(certFile, keyFile, serialFile, name, lifetime)
	return ca, true, err
}

// if serialFile is empty, a RandomSerialGenerator will be used
func GetCA(certFile, keyFile, serialFile string) (*CA, error) {
	caConfig, err := GetTLSCertificateConfig(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	var serialGenerator SerialGenerator
	if len(serialFile) > 0 {
		serialGenerator, err = NewSerialFileGenerator(serialFile)
		if err != nil {
			return nil, err
		}
	} else {
		serialGenerator = &RandomSerialGenerator{}
	}

	return &CA{
		SerialGenerator: serialGenerator,
		Config:          caConfig,
	}, nil
}

func GetCAFromBytes(certBytes, keyBytes []byte) (*CA, error) {
	caConfig, err := GetTLSCertificateConfigFromBytes(certBytes, keyBytes)
	if err != nil {
		return nil, err
	}

	return &CA{
		SerialGenerator: &RandomSerialGenerator{},
		Config:          caConfig,
	}, nil
}

// if serialFile is empty, a RandomSerialGenerator will be used
func MakeSelfSignedCA(certFile, keyFile, serialFile, name string, lifetime time.Duration) (*CA, error) {
	klog.V(2).Infof("Generating new CA for %s cert, and key in %s, %s", name, certFile, keyFile)

	caConfig, err := MakeSelfSignedCAConfig(name, lifetime)
	if err != nil {
		return nil, err
	}
	if err := caConfig.WriteCertConfigFile(certFile, keyFile); err != nil {
		return nil, err
	}

	var serialGenerator SerialGenerator
	if len(serialFile) > 0 {
		// create / overwrite the serial file with a zero padded hex value (ending in a newline to have a valid file)
		if err := os.WriteFile(serialFile, []byte("00\n"), 0644); err != nil {
			return nil, err
		}
		serialGenerator, err = NewSerialFileGenerator(serialFile)
		if err != nil {
			return nil, err
		}
	} else {
		serialGenerator = &RandomSerialGenerator{}
	}

	return &CA{
		SerialGenerator: serialGenerator,
		Config:          caConfig,
	}, nil
}

func MakeSelfSignedCAConfig(name string, lifetime time.Duration) (*TLSCertificateConfig, error) {
	subject := pkix.Name{CommonName: name}
	return MakeSelfSignedCAConfigForSubject(subject, lifetime)
}

func MakeSelfSignedCAConfigForSubject(subject pkix.Name, lifetime time.Duration) (*TLSCertificateConfig, error) {
	if lifetime <= 0 {
		lifetime = DefaultCACertificateLifetimeDuration
		fmt.Fprintf(os.Stderr, "Validity period of the certificate for %q is unset, resetting to %s!\n", subject.CommonName, lifetime.String())
	}

	if lifetime > DefaultCACertificateLifetimeDuration {
		warnAboutCertificateLifeTime(subject.CommonName, DefaultCACertificateLifetimeDuration)
	}
	return makeSelfSignedCAConfigForSubjectAndDuration(subject, time.Now, lifetime)
}

func MakeSelfSignedCAConfigForDuration(name string, caLifetime time.Duration) (*TLSCertificateConfig, error) {
	subject := pkix.Name{CommonName: name}
	return makeSelfSignedCAConfigForSubjectAndDuration(subject, time.Now, caLifetime)
}

func UnsafeMakeSelfSignedCAConfigForDurationAtTime(name string, currentTime func() time.Time, caLifetime time.Duration) (*TLSCertificateConfig, error) {
	subject := pkix.Name{CommonName: name}
	return makeSelfSignedCAConfigForSubjectAndDuration(subject, currentTime, caLifetime)
}

func makeSelfSignedCAConfigForSubjectAndDuration(subject pkix.Name, currentTime func() time.Time, caLifetime time.Duration) (*TLSCertificateConfig, error) {
	// Create CA cert
	rootcaPublicKey, rootcaPrivateKey, publicKeyHash, err := newKeyPairWithHash()
	if err != nil {
		return nil, err
	}
	// AuthorityKeyId and SubjectKeyId should match for a self-signed CA
	authorityKeyId := publicKeyHash
	subjectKeyId := publicKeyHash
	rootcaTemplate := newSigningCertificateTemplateForDuration(subject, caLifetime, currentTime, authorityKeyId, subjectKeyId)
	rootcaCert, err := signCertificate(rootcaTemplate, rootcaPublicKey, rootcaTemplate, rootcaPrivateKey)
	if err != nil {
		return nil, err
	}
	caConfig := &TLSCertificateConfig{
		Certs: []*x509.Certificate{rootcaCert},
		Key:   rootcaPrivateKey,
	}
	return caConfig, nil
}

func MakeCAConfigForDuration(name string, caLifetime time.Duration, issuer *CA) (*TLSCertificateConfig, error) {
	// Create CA cert
	signerPublicKey, signerPrivateKey, publicKeyHash, err := newKeyPairWithHash()
	if err != nil {
		return nil, err
	}
	authorityKeyId := issuer.Config.Certs[0].SubjectKeyId
	subjectKeyId := publicKeyHash
	signerTemplate := newSigningCertificateTemplateForDuration(pkix.Name{CommonName: name}, caLifetime, time.Now, authorityKeyId, subjectKeyId)
	signerCert, err := issuer.SignCertificate(signerTemplate, signerPublicKey)
	if err != nil {
		return nil, err
	}
	signerConfig := &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{signerCert}, issuer.Config.Certs...),
		Key:   signerPrivateKey,
	}
	return signerConfig, nil
}

// EnsureSubCA returns a subCA signed by the `ca`, whether it was created
// (as opposed to pre-existing), and any error that might occur during the subCA
// creation.
// If serialFile is an empty string, a RandomSerialGenerator will be used.
func (ca *CA) EnsureSubCA(certFile, keyFile, serialFile, name string, lifetime time.Duration) (*CA, bool, error) {
	if subCA, err := GetCA(certFile, keyFile, serialFile); err == nil {
		return subCA, false, err
	}
	subCA, err := ca.MakeAndWriteSubCA(certFile, keyFile, serialFile, name, lifetime)
	return subCA, true, err
}

// MakeAndWriteSubCA returns a new sub-CA configuration. New cert/key pair is generated
// while using this function.
// If serialFile is an empty string, a RandomSerialGenerator will be used.
func (ca *CA) MakeAndWriteSubCA(certFile, keyFile, serialFile, name string, lifetime time.Duration) (*CA, error) {
	klog.V(4).Infof("Generating sub-CA certificate in %s, key in %s, serial in %s", certFile, keyFile, serialFile)

	subCAConfig, err := MakeCAConfigForDuration(name, lifetime, ca)
	if err != nil {
		return nil, err
	}

	if err := subCAConfig.WriteCertConfigFile(certFile, keyFile); err != nil {
		return nil, err
	}

	var serialGenerator SerialGenerator
	if len(serialFile) > 0 {
		// create / overwrite the serial file with a zero padded hex value (ending in a newline to have a valid file)
		if err := os.WriteFile(serialFile, []byte("00\n"), 0644); err != nil {
			return nil, err
		}

		serialGenerator, err = NewSerialFileGenerator(serialFile)
		if err != nil {
			return nil, err
		}
	} else {
		serialGenerator = &RandomSerialGenerator{}
	}

	return &CA{
		Config:          subCAConfig,
		SerialGenerator: serialGenerator,
	}, nil
}

func (ca *CA) EnsureServerCert(certFile, keyFile string, hostnames sets.Set[string], lifetime time.Duration) (*TLSCertificateConfig, bool, error) {
	certConfig, err := GetServerCert(certFile, keyFile, hostnames)
	if err != nil {
		certConfig, err = ca.MakeAndWriteServerCert(certFile, keyFile, hostnames, lifetime)
		return certConfig, true, err
	}

	return certConfig, false, nil
}

func GetServerCert(certFile, keyFile string, hostnames sets.Set[string]) (*TLSCertificateConfig, error) {
	server, err := GetTLSCertificateConfig(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	cert := server.Certs[0]
	certNames := sets.New[string]()
	for _, ip := range cert.IPAddresses {
		certNames.Insert(ip.String())
	}
	certNames.Insert(cert.DNSNames...)
	if hostnames.Equal(certNames) {
		klog.V(4).Infof("Found existing server certificate in %s", certFile)
		return server, nil
	}

	return nil, fmt.Errorf("existing server certificate in %s does not match required hostnames", certFile)
}

func (ca *CA) MakeAndWriteServerCert(certFile, keyFile string, hostnames sets.Set[string], lifetime time.Duration) (*TLSCertificateConfig, error) {
	klog.V(4).Infof("Generating server certificate in %s, key in %s", certFile, keyFile)

	server, err := ca.MakeServerCert(hostnames, lifetime)
	if err != nil {
		return nil, err
	}
	if err := server.WriteCertConfigFile(certFile, keyFile); err != nil {
		return server, err
	}
	return server, nil
}

// CertificateExtensionFunc is passed a certificate that it may extend, or return an error
// if the extension attempt failed.
type CertificateExtensionFunc func(*x509.Certificate) error

func (ca *CA) MakeServerCert(hostnames sets.Set[string], lifetime time.Duration, fns ...CertificateExtensionFunc) (*TLSCertificateConfig, error) {
	serverPublicKey, serverPrivateKey, publicKeyHash, _ := newKeyPairWithHash()
	authorityKeyId := ca.Config.Certs[0].SubjectKeyId
	subjectKeyId := publicKeyHash
	serverTemplate := newServerCertificateTemplate(pkix.Name{CommonName: sets.List(hostnames)[0]}, sets.List(hostnames), lifetime, time.Now, authorityKeyId, subjectKeyId)
	for _, fn := range fns {
		if err := fn(serverTemplate); err != nil {
			return nil, err
		}
	}
	serverCrt, err := ca.SignCertificate(serverTemplate, serverPublicKey)
	if err != nil {
		return nil, err
	}
	server := &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{serverCrt}, ca.Config.Certs...),
		Key:   serverPrivateKey,
	}
	return server, nil
}

func (ca *CA) MakeServerCertForDuration(hostnames sets.Set[string], lifetime time.Duration, fns ...CertificateExtensionFunc) (*TLSCertificateConfig, error) {
	serverPublicKey, serverPrivateKey, publicKeyHash, _ := newKeyPairWithHash()
	authorityKeyId := ca.Config.Certs[0].SubjectKeyId
	subjectKeyId := publicKeyHash
	serverTemplate := newServerCertificateTemplateForDuration(pkix.Name{CommonName: sets.List(hostnames)[0]}, sets.List(hostnames), lifetime, time.Now, authorityKeyId, subjectKeyId)
	for _, fn := range fns {
		if err := fn(serverTemplate); err != nil {
			return nil, err
		}
	}
	serverCrt, err := ca.SignCertificate(serverTemplate, serverPublicKey)
	if err != nil {
		return nil, err
	}
	server := &TLSCertificateConfig{
		Certs: append([]*x509.Certificate{serverCrt}, ca.Config.Certs...),
		Key:   serverPrivateKey,
	}
	return server, nil
}

func (ca *CA) EnsureClientCertificate(certFile, keyFile string, u user.Info, lifetime time.Duration) (*TLSCertificateConfig, bool, error) {
	certConfig, err := GetClientCertificate(certFile, keyFile, u)
	if err != nil {
		certConfig, err = ca.MakeClientCertificate(certFile, keyFile, u, lifetime)
		return certConfig, true, err // true indicates we wrote the files.
	}
	return certConfig, false, nil
}

func GetClientCertificate(certFile, keyFile string, u user.Info) (*TLSCertificateConfig, error) {
	certConfig, err := GetTLSCertificateConfig(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	if subject := certConfig.Certs[0].Subject; subjectChanged(subject, UserToSubject(u)) {
		return nil, fmt.Errorf("existing client certificate in %s was issued for a different Subject (%s)",
			certFile, subject)
	}

	return certConfig, nil
}

func subjectChanged(existing, expected pkix.Name) bool {
	sort.Strings(existing.Organization)
	sort.Strings(expected.Organization)

	return existing.CommonName != expected.CommonName ||
		existing.SerialNumber != expected.SerialNumber ||
		!reflect.DeepEqual(existing.Organization, expected.Organization)
}

func (ca *CA) MakeClientCertificate(certFile, keyFile string, u user.Info, lifetime time.Duration) (*TLSCertificateConfig, error) {
	klog.V(4).Infof("Generating client cert in %s and key in %s", certFile, keyFile)
	// ensure parent dirs
	if err := os.MkdirAll(filepath.Dir(certFile), os.FileMode(0755)); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(keyFile), os.FileMode(0755)); err != nil {
		return nil, err
	}

	clientPublicKey, clientPrivateKey, _ := NewKeyPair()
	clientTemplate := NewClientCertificateTemplate(UserToSubject(u), lifetime, time.Now)
	clientCrt, err := ca.SignCertificate(clientTemplate, clientPublicKey)
	if err != nil {
		return nil, err
	}

	certData, err := EncodeCertificates(clientCrt)
	if err != nil {
		return nil, err
	}
	keyData, err := EncodeKey(clientPrivateKey)
	if err != nil {
		return nil, err
	}

	if err = os.WriteFile(certFile, certData, os.FileMode(0644)); err != nil {
		return nil, err
	}
	if err = os.WriteFile(keyFile, keyData, os.FileMode(0600)); err != nil {
		return nil, err
	}

	return GetTLSCertificateConfig(certFile, keyFile)
}

func (ca *CA) MakeClientCertificateForDuration(u user.Info, lifetime time.Duration) (*TLSCertificateConfig, error) {
	clientPublicKey, clientPrivateKey, _ := NewKeyPair()
	clientTemplate := NewClientCertificateTemplateForDuration(UserToSubject(u), lifetime, time.Now)
	clientCrt, err := ca.SignCertificate(clientTemplate, clientPublicKey)
	if err != nil {
		return nil, err
	}

	certData, err := EncodeCertificates(clientCrt)
	if err != nil {
		return nil, err
	}
	keyData, err := EncodeKey(clientPrivateKey)
	if err != nil {
		return nil, err
	}

	return GetTLSCertificateConfigFromBytes(certData, keyData)
}

type sortedForDER []string

func (s sortedForDER) Len() int {
	return len(s)
}
func (s sortedForDER) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortedForDER) Less(i, j int) bool {
	l1 := len(s[i])
	l2 := len(s[j])
	if l1 == l2 {
		return s[i] < s[j]
	}
	return l1 < l2
}

func UserToSubject(u user.Info) pkix.Name {
	// Ok we are going to order groups in a peculiar way here to workaround a
	// 2 bugs, 1 in golang (https://github.com/golang/go/issues/24254) which
	// incorrectly encodes Multivalued RDNs and another in GNUTLS clients
	// which are too picky (https://gitlab.com/gnutls/gnutls/issues/403)
	// and try to "correct" this issue when reading client certs.
	//
	// This workaround should be killed once Golang's pkix module is fixed to
	// generate a correct DER encoding.
	//
	// The workaround relies on the fact that the first octect that differs
	// between the encoding of two group RDNs will end up being the encoded
	// length which is directly related to the group name's length. So we'll
	// sort such that shortest names come first.
	ugroups := u.GetGroups()
	groups := make([]string, len(ugroups))
	copy(groups, ugroups)
	sort.Sort(sortedForDER(groups))

	return pkix.Name{
		CommonName:   u.GetName(),
		SerialNumber: u.GetUID(),
		Organization: groups,
	}
}

func (ca *CA) SignCertificate(template *x509.Certificate, requestKey crypto.PublicKey) (*x509.Certificate, error) {
	// Increment and persist serial
	serial, err := ca.SerialGenerator.Next(template)
	if err != nil {
		return nil, err
	}
	template.SerialNumber = big.NewInt(serial)
	return signCertificate(template, requestKey, ca.Config.Certs[0], ca.Config.Key)
}

func NewKeyPair() (crypto.PublicKey, crypto.PrivateKey, error) {
	return newRSAKeyPair()
}

func newKeyPairWithHash() (crypto.PublicKey, crypto.PrivateKey, []byte, error) {
	publicKey, privateKey, err := newRSAKeyPair()
	var publicKeyHash []byte
	if err == nil {
		hash := sha1.New()
		hash.Write(publicKey.N.Bytes())
		publicKeyHash = hash.Sum(nil)
	}
	return publicKey, privateKey, publicKeyHash, err
}

func newRSAKeyPair() (*rsa.PublicKey, *rsa.PrivateKey, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, keyBits)
	if err != nil {
		return nil, nil, err
	}
	return &privateKey.PublicKey, privateKey, nil
}

// Can be used for CA or intermediate signing certs
func newSigningCertificateTemplateForDuration(subject pkix.Name, caLifetime time.Duration, currentTime func() time.Time, authorityKeyId, subjectKeyId []byte) *x509.Certificate {
	return &x509.Certificate{
		Subject: subject,

		SignatureAlgorithm: x509.SHA256WithRSA,

		NotBefore: currentTime().Add(-1 * time.Second),
		NotAfter:  currentTime().Add(caLifetime),

		// Specify a random serial number to avoid the same issuer+serial
		// number referring to different certs in a chain of trust if the
		// signing certificate is ever rotated.
		SerialNumber: big.NewInt(randomSerialNumber()),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,

		AuthorityKeyId: authorityKeyId,
		SubjectKeyId:   subjectKeyId,
	}
}

// Can be used for ListenAndServeTLS
func newServerCertificateTemplate(subject pkix.Name, hosts []string, lifetime time.Duration, currentTime func() time.Time, authorityKeyId, subjectKeyId []byte) *x509.Certificate {
	if lifetime <= 0 {
		lifetime = DefaultCertificateLifetimeDuration
		fmt.Fprintf(os.Stderr, "Validity period of the certificate for %q is unset, resetting to %s!\n", subject.CommonName, lifetime.String())
	}

	if lifetime > DefaultCertificateLifetimeDuration {
		warnAboutCertificateLifeTime(subject.CommonName, DefaultCertificateLifetimeDuration)
	}

	return newServerCertificateTemplateForDuration(subject, hosts, lifetime, currentTime, authorityKeyId, subjectKeyId)
}

// Can be used for ListenAndServeTLS
func newServerCertificateTemplateForDuration(subject pkix.Name, hosts []string, lifetime time.Duration, currentTime func() time.Time, authorityKeyId, subjectKeyId []byte) *x509.Certificate {
	template := &x509.Certificate{
		Subject: subject,

		SignatureAlgorithm: x509.SHA256WithRSA,

		NotBefore:    currentTime().Add(-1 * time.Second),
		NotAfter:     currentTime().Add(lifetime),
		SerialNumber: big.NewInt(1),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,

		AuthorityKeyId: authorityKeyId,
		SubjectKeyId:   subjectKeyId,
	}

	template.IPAddresses, template.DNSNames = IPAddressesDNSNames(hosts)

	return template
}

func IPAddressesDNSNames(hosts []string) ([]net.IP, []string) {
	ips := []net.IP{}
	dns := []string{}
	for _, host := range hosts {
		if ip := net.ParseIP(host); ip != nil {
			ips = append(ips, ip)
		} else {
			dns = append(dns, host)
		}
	}

	// Include IP addresses as DNS subjectAltNames in the cert as well, for the sake of Python, Windows (< 10), and unnamed other libraries
	// Ensure these technically invalid DNS subjectAltNames occur after the valid ones, to avoid triggering cert errors in Firefox
	// See https://bugzilla.mozilla.org/show_bug.cgi?id=1148766
	for _, ip := range ips {
		dns = append(dns, ip.String())
	}

	return ips, dns
}

func CertsFromPEM(pemCerts []byte) ([]*x509.Certificate, error) {
	ok := false
	certs := []*x509.Certificate{}
	for len(pemCerts) > 0 {
		var block *pem.Block
		block, pemCerts = pem.Decode(pemCerts)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return certs, err
		}

		certs = append(certs, cert)
		ok = true
	}

	if !ok {
		return certs, errors.New("could not read any certificates")
	}
	return certs, nil
}

// Can be used as a certificate in http.Transport TLSClientConfig
func NewClientCertificateTemplate(subject pkix.Name, lifetime time.Duration, currentTime func() time.Time) *x509.Certificate {
	if lifetime <= 0 {
		lifetime = DefaultCertificateLifetimeDuration
		fmt.Fprintf(os.Stderr, "Validity period of the certificate for %q is unset, resetting to %s!\n", subject.CommonName, lifetime.String())
	}

	if lifetime > DefaultCertificateLifetimeDuration {
		warnAboutCertificateLifeTime(subject.CommonName, DefaultCertificateLifetimeDuration)
	}

	return NewClientCertificateTemplateForDuration(subject, lifetime, currentTime)
}

// Can be used as a certificate in http.Transport TLSClientConfig
func NewClientCertificateTemplateForDuration(subject pkix.Name, lifetime time.Duration, currentTime func() time.Time) *x509.Certificate {
	return &x509.Certificate{
		Subject: subject,

		SignatureAlgorithm: x509.SHA256WithRSA,

		NotBefore:    currentTime().Add(-1 * time.Second),
		NotAfter:     currentTime().Add(lifetime),
		SerialNumber: big.NewInt(1),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
}

func warnAboutCertificateLifeTime(name string, defaultLifetimeDuration time.Duration) {
	defaultLifetimeInYears := defaultLifetimeDuration / 365 / 24
	fmt.Fprintf(os.Stderr, "WARNING: Validity period of the certificate for %q is greater than %d years!\n", name, defaultLifetimeInYears)
	fmt.Fprintln(os.Stderr, "WARNING: By security reasons it is strongly recommended to change this period and make it smaller!")
}

func signCertificate(template *x509.Certificate, requestKey crypto.PublicKey, issuer *x509.Certificate, issuerKey crypto.PrivateKey) (*x509.Certificate, error) {
	derBytes, err := x509.CreateCertificate(rand.Reader, template, issuer, requestKey, issuerKey)
	if err != nil {
		return nil, err
	}
	certs, err := x509.ParseCertificates(derBytes)
	if err != nil {
		return nil, err
	}
	if len(certs) != 1 {
		return nil, errors.New("expected a single certificate")
	}
	return certs[0], nil
}

func EncodeCertificates(certs ...*x509.Certificate) ([]byte, error) {
	b := bytes.Buffer{}
	for _, cert := range certs {
		if err := pem.Encode(&b, &pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw}); err != nil {
			return []byte{}, err
		}
	}
	return b.Bytes(), nil
}
func EncodeKey(key crypto.PrivateKey) ([]byte, error) {
	b := bytes.Buffer{}
	switch key := key.(type) {
	case *ecdsa.PrivateKey:
		keyBytes, err := x509.MarshalECPrivateKey(key)
		if err != nil {
			return []byte{}, err
		}
		if err := pem.Encode(&b, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}); err != nil {
			return b.Bytes(), err
		}
	case *rsa.PrivateKey:
		if err := pem.Encode(&b, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)}); err != nil {
			return []byte{}, err
		}
	default:
		return []byte{}, errors.New("unrecognized key type")

	}
	return b.Bytes(), nil
}

func writeCertificates(f io.Writer, certs ...*x509.Certificate) error {
	bytes, err := EncodeCertificates(certs...)
	if err != nil {
		return err
	}
	if _, err := f.Write(bytes); err != nil {
		return err
	}

	return nil
}
func writeKeyFile(f io.Writer, key crypto.PrivateKey) error {
	bytes, err := EncodeKey(key)
	if err != nil {
		return err
	}
	if _, err := f.Write(bytes); err != nil {
		return err
	}

	return nil
}
