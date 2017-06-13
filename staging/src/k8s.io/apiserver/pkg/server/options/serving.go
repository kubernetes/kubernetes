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

package options

import (
	"crypto/tls"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net"
	"path"
	"strconv"
	"time"

	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"github.com/spf13/pflag"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/server"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	certutil "k8s.io/client-go/util/cert"
)

type SecureServingOptions struct {
	BindAddress net.IP
	BindPort    int

	// ServerCert is the TLS cert info for serving secure traffic
	ServerCert GeneratableKeyCert
	// SNICertKeys are named CertKeys for serving secure traffic with SNI support.
	SNICertKeys []utilflag.NamedCertKey

	// when set determines whether to use loopback configuration to create shared informers.
	useLoopbackCfg bool
}

type CertKey struct {
	// CertFile is a file containing a PEM-encoded certificate, and possibly the complete certificate chain
	CertFile string
	// KeyFile is a file containing a PEM-encoded private key for the certificate specified by CertFile
	KeyFile string
}

type GeneratableKeyCert struct {
	CertKey CertKey

	// CACertFile is an optional file containing the certificate chain for CertKey.CertFile
	CACertFile string
	// CertDirectory is a directory that will contain the certificates.  If the cert and key aren't specifically set
	// this will be used to derive a match with the "pair-name"
	CertDirectory string
	// PairName is the name which will be used with CertDirectory to make a cert and key names
	// It becomes CertDirector/PairName.crt and CertDirector/PairName.key
	PairName string
}

func NewSecureServingOptions() *SecureServingOptions {
	return &SecureServingOptions{
		BindAddress: net.ParseIP("0.0.0.0"),
		BindPort:    443,
		ServerCert: GeneratableKeyCert{
			PairName:      "apiserver",
			CertDirectory: "apiserver.local.config/certificates",
		},
	}
}

func (s *SecureServingOptions) DefaultExternalAddress() (net.IP, error) {
	return utilnet.ChooseBindAddress(s.BindAddress)
}

func (s *SecureServingOptions) Validate() []error {
	errors := []error{}

	if s.BindPort < 0 || s.BindPort > 65535 {
		errors = append(errors, fmt.Errorf("--secure-port %v must be between 0 and 65535, inclusive. 0 for turning off secure port.", s.BindPort))
	}

	return errors
}

func (s *SecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "bind-address", s.BindAddress, ""+
		"The IP address on which to listen for the --secure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		"clients. If blank, all interfaces will be used (0.0.0.0).")

	fs.IntVar(&s.BindPort, "secure-port", s.BindPort, ""+
		"The port on which to serve HTTPS with authentication and authorization. If 0, "+
		"don't serve HTTPS at all.")

	fs.StringVar(&s.ServerCert.CertDirectory, "cert-dir", s.ServerCert.CertDirectory, ""+
		"The directory where the TLS certs are located. "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")

	fs.StringVar(&s.ServerCert.CertKey.CertFile, "tls-cert-file", s.ServerCert.CertKey.CertFile, ""+
		"File containing the default x509 Certificate for HTTPS. (CA cert, if any, concatenated "+
		"after server cert). If HTTPS serving is enabled, and --tls-cert-file and "+
		"--tls-private-key-file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to /var/run/kubernetes.")

	fs.StringVar(&s.ServerCert.CertKey.KeyFile, "tls-private-key-file", s.ServerCert.CertKey.KeyFile,
		"File containing the default x509 private key matching --tls-cert-file.")

	fs.StringVar(&s.ServerCert.CACertFile, "tls-ca-file", s.ServerCert.CACertFile, "If set, this "+
		"certificate authority will used for secure access from Admission "+
		"Controllers. This must be a valid PEM-encoded CA bundle. Altneratively, the certificate authority "+
		"can be appended to the certificate provided by --tls-cert-file.")

	fs.Var(utilflag.NewNamedCertKeyArray(&s.SNICertKeys), "tls-sni-cert-key", ""+
		"A pair of x509 certificate and private key file paths, optionally suffixed with a list of "+
		"domain patterns which are fully qualified domain names, possibly with prefixed wildcard "+
		"segments. If no domain patterns are provided, the names of the certificate are "+
		"extracted. Non-wildcard matches trump over wildcard matches, explicit domain patterns "+
		"trump over extracted names. For multiple key/certificate pairs, use the "+
		"--tls-sni-cert-key multiple times. "+
		"Examples: \"example.crt,example.key\" or \"foo.crt,foo.key:*.foo.com,foo.com\".")
}

func (s *SecureServingOptions) AddDeprecatedFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "public-address-override", s.BindAddress,
		"DEPRECATED: see --bind-address instead.")
	fs.MarkDeprecated("public-address-override", "see --bind-address instead.")
}

// ApplyTo fills up serving information in the server configuration.
func (s *SecureServingOptions) ApplyTo(c *server.Config) error {
	if s.BindPort <= 0 {
		return nil
	}
	if err := s.applyServingInfoTo(c); err != nil {
		return err
	}

	// create self-signed cert+key with the fake server.LoopbackClientServerNameOverride and
	// let the server return it when the loopback client connects.
	certPem, keyPem, err := certutil.GenerateSelfSignedCertKey(server.LoopbackClientServerNameOverride, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}
	tlsCert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}

	secureLoopbackClientConfig, err := c.SecureServingInfo.NewLoopbackClientConfig(uuid.NewRandom().String(), certPem)
	switch {
	// if we failed and there's no fallback loopback client config, we need to fail
	case err != nil && c.LoopbackClientConfig == nil:
		return err

	// if we failed, but we already have a fallback loopback client config (usually insecure), allow it
	case err != nil && c.LoopbackClientConfig != nil:

	default:
		c.LoopbackClientConfig = secureLoopbackClientConfig
		c.SecureServingInfo.SNICerts[server.LoopbackClientServerNameOverride] = &tlsCert
	}

	// create shared informers, if not explicitly set use in cluster config.
	// do not fail on an error, this allows an external API server to startup
	// outside of a kube cluster.
	var clientCfg *rest.Config
	err = nil
	if s.useLoopbackCfg {
		clientCfg = c.LoopbackClientConfig
	} else {
		clientCfg, err = rest.InClusterConfig()
	}
	if err != nil {
		glog.Errorf("Couldn't create in cluster config due to %v. SharedInformerFactory will not be set.", err)
		return nil
	}
	clientset, err := kubernetes.NewForConfig(clientCfg)
	if err != nil {
		glog.Errorf("Couldn't create clientset due to %v. SharedInformerFactory will not be set.", err)
		return nil
	}
	c.SharedInformerFactory = informers.NewSharedInformerFactory(clientset, 10*time.Minute)
	return nil
}

// ForceLoopbackConfigUsage forces the usage of the loopback configuration
// to create SharedInformerFactory. The primary client of this method
// is kube API server, no other API server is the source of truth for kube APIs.
//
// Note:
// this method MUST be called prior to ApplyTo to take an effect.
func (s *SecureServingOptions) ForceLoopbackConfigUsage() {
	s.useLoopbackCfg = true
}

func (s *SecureServingOptions) applyServingInfoTo(c *server.Config) error {
	if s.BindPort <= 0 {
		return nil
	}

	secureServingInfo := &server.SecureServingInfo{
		BindAddress: net.JoinHostPort(s.BindAddress.String(), strconv.Itoa(s.BindPort)),
	}

	serverCertFile, serverKeyFile := s.ServerCert.CertKey.CertFile, s.ServerCert.CertKey.KeyFile

	// load main cert
	if len(serverCertFile) != 0 || len(serverKeyFile) != 0 {
		tlsCert, err := tls.LoadX509KeyPair(serverCertFile, serverKeyFile)
		if err != nil {
			return fmt.Errorf("unable to load server certificate: %v", err)
		}
		secureServingInfo.Cert = &tlsCert
	}

	// optionally load CA cert
	if len(s.ServerCert.CACertFile) != 0 {
		pemData, err := ioutil.ReadFile(s.ServerCert.CACertFile)
		if err != nil {
			return fmt.Errorf("failed to read certificate authority from %q: %v", s.ServerCert.CACertFile, err)
		}
		block, pemData := pem.Decode(pemData)
		if block == nil {
			return fmt.Errorf("no certificate found in certificate authority file %q", s.ServerCert.CACertFile)
		}
		if block.Type != "CERTIFICATE" {
			return fmt.Errorf("expected CERTIFICATE block in certiticate authority file %q, found: %s", s.ServerCert.CACertFile, block.Type)
		}
		secureServingInfo.CACert = &tls.Certificate{
			Certificate: [][]byte{block.Bytes},
		}
	}

	// load SNI certs
	namedTLSCerts := make([]server.NamedTLSCert, 0, len(s.SNICertKeys))
	for _, nck := range s.SNICertKeys {
		tlsCert, err := tls.LoadX509KeyPair(nck.CertFile, nck.KeyFile)
		namedTLSCerts = append(namedTLSCerts, server.NamedTLSCert{
			TLSCert: tlsCert,
			Names:   nck.Names,
		})
		if err != nil {
			return fmt.Errorf("failed to load SNI cert and key: %v", err)
		}
	}
	var err error
	secureServingInfo.SNICerts, err = server.GetNamedCertificateMap(namedTLSCerts)
	if err != nil {
		return err
	}

	c.SecureServingInfo = secureServingInfo
	c.ReadWritePort = s.BindPort

	return nil
}

func (s *SecureServingOptions) MaybeDefaultWithSelfSignedCerts(publicAddress string, alternateDNS []string, alternateIPs []net.IP) error {
	if s == nil {
		return nil
	}
	keyCert := &s.ServerCert.CertKey
	if s.BindPort == 0 || len(keyCert.CertFile) != 0 || len(keyCert.KeyFile) != 0 {
		return nil
	}

	keyCert.CertFile = path.Join(s.ServerCert.CertDirectory, s.ServerCert.PairName+".crt")
	keyCert.KeyFile = path.Join(s.ServerCert.CertDirectory, s.ServerCert.PairName+".key")

	canReadCertAndKey, err := certutil.CanReadCertAndKey(keyCert.CertFile, keyCert.KeyFile)
	if err != nil {
		return err
	}
	if !canReadCertAndKey {
		// add either the bind address or localhost to the valid alternates
		bindIP := s.BindAddress.String()
		if bindIP == "0.0.0.0" {
			alternateDNS = append(alternateDNS, "localhost")
		} else {
			alternateIPs = append(alternateIPs, s.BindAddress)
		}

		if cert, key, err := certutil.GenerateSelfSignedCertKey(publicAddress, alternateIPs, alternateDNS); err != nil {
			return fmt.Errorf("unable to generate self signed cert: %v", err)
		} else {
			if err := certutil.WriteCert(keyCert.CertFile, cert); err != nil {
				return err
			}

			if err := certutil.WriteKey(keyCert.KeyFile, key); err != nil {
				return err
			}
			glog.Infof("Generated self-signed cert (%s, %s)", keyCert.CertFile, keyCert.KeyFile)
		}
	}

	return nil
}
