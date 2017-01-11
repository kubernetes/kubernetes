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
	"fmt"
	"net"
	"path"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/util/config"
)

type ServingOptions struct {
	BindAddress net.IP
	BindPort    int
}

type SecureServingOptions struct {
	ServingOptions ServingOptions

	// ServerCert is the TLS cert info for serving secure traffic
	ServerCert GeneratableKeyCert
	// SNICertKeys are named CertKeys for serving secure traffic with SNI support.
	SNICertKeys []config.NamedCertKey
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
		ServingOptions: ServingOptions{
			BindAddress: net.ParseIP("0.0.0.0"),
			BindPort:    6443,
		},
		ServerCert: GeneratableKeyCert{
			PairName:      "apiserver",
			CertDirectory: "/var/run/kubernetes",
		},
	}
}

func (s *SecureServingOptions) Validate() []error {
	errors := []error{}
	if s == nil {
		return errors
	}

	errors = append(errors, s.ServingOptions.Validate("secure-port")...)
	return errors
}

func (s *SecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.ServingOptions.BindAddress, "bind-address", s.ServingOptions.BindAddress, ""+
		"The IP address on which to listen for the --secure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		"clients. If blank, all interfaces will be used (0.0.0.0).")

	fs.IntVar(&s.ServingOptions.BindPort, "secure-port", s.ServingOptions.BindPort, ""+
		"The port on which to serve HTTPS with authentication and authorization. If 0, "+
		"don't serve HTTPS at all.")

	fs.StringVar(&s.ServerCert.CertDirectory, "cert-dir", s.ServerCert.CertDirectory, ""+
		"The directory where the TLS certs are located (by default /var/run/kubernetes). "+
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

	fs.Var(config.NewNamedCertKeyArray(&s.SNICertKeys), "tls-sni-cert-key", ""+
		"A pair of x509 certificate and private key file paths, optionally suffixed with a list of "+
		"domain patterns which are fully qualified domain names, possibly with prefixed wildcard "+
		"segments. If no domain patterns are provided, the names of the certificate are "+
		"extracted. Non-wildcard matches trump over wildcard matches, explicit domain patterns "+
		"trump over extracted names. For multiple key/certificate pairs, use the "+
		"--tls-sni-cert-key multiple times. "+
		"Examples: \"example.key,example.crt\" or \"*.foo.com,foo.com:foo.key,foo.crt\".")
}

func (s *SecureServingOptions) AddDeprecatedFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.ServingOptions.BindAddress, "public-address-override", s.ServingOptions.BindAddress,
		"DEPRECATED: see --bind-address instead.")
	fs.MarkDeprecated("public-address-override", "see --bind-address instead.")
}

func NewInsecureServingOptions() *ServingOptions {
	return &ServingOptions{
		BindAddress: net.ParseIP("127.0.0.1"),
		BindPort:    8080,
	}
}

func (s ServingOptions) Validate(portArg string) []error {
	errors := []error{}

	if s.BindPort < 0 || s.BindPort > 65535 {
		errors = append(errors, fmt.Errorf("--%v %v must be between 0 and 65535, inclusive. 0 for turning off secure port.", portArg, s.BindPort))
	}

	return errors
}

func (s *ServingOptions) DefaultExternalAddress() (net.IP, error) {
	return utilnet.ChooseBindAddress(s.BindAddress)
}

func (s *ServingOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "insecure-bind-address", s.BindAddress, ""+
		"The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). "+
		"Defaults to localhost.")

	fs.IntVar(&s.BindPort, "insecure-port", s.BindPort, ""+
		"The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed "+
		"that firewall rules are set up such that this port is not reachable from outside of "+
		"the cluster and that port 443 on the cluster's public address is proxied to this "+
		"port. This is performed by nginx in the default setup.")
}

func (s *ServingOptions) AddDeprecatedFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "address", s.BindAddress,
		"DEPRECATED: see --insecure-bind-address instead.")
	fs.MarkDeprecated("address", "see --insecure-bind-address instead.")

	fs.IntVar(&s.BindPort, "port", s.BindPort, "DEPRECATED: see --insecure-port instead.")
	fs.MarkDeprecated("port", "see --insecure-port instead.")
}

func (s *SecureServingOptions) MaybeDefaultWithSelfSignedCerts(publicAddress string, alternateIPs ...net.IP) error {
	if s == nil {
		return nil
	}
	keyCert := &s.ServerCert.CertKey
	if s.ServingOptions.BindPort == 0 || len(keyCert.CertFile) != 0 || len(keyCert.KeyFile) != 0 {
		return nil
	}

	keyCert.CertFile = path.Join(s.ServerCert.CertDirectory, s.ServerCert.PairName+".crt")
	keyCert.KeyFile = path.Join(s.ServerCert.CertDirectory, s.ServerCert.PairName+".key")

	canReadCertAndKey, err := certutil.CanReadCertAndKey(keyCert.CertFile, keyCert.KeyFile)
	if err != nil {
		return err
	}
	if !canReadCertAndKey {
		// TODO: It would be nice to set a fqdn subject alt name, but only the kubelets know, the apiserver is clueless
		// alternateDNS = append(alternateDNS, "kubernetes.default.svc.CLUSTER.DNS.NAME")
		// TODO (cjcullen): Is ClusterIP the right address to sign a cert with?
		alternateDNS := []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}

		// add either the bind address or localhost to the valid alternates
		bindIP := s.ServingOptions.BindAddress.String()
		if bindIP == "0.0.0.0" {
			alternateDNS = append(alternateDNS, "localhost")
		} else {
			alternateIPs = append(alternateIPs, s.ServingOptions.BindAddress)
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
