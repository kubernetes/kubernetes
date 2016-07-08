package utils

import (
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"os"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gofig"
	"github.com/akutz/goof"
	"github.com/akutz/gotil"

	"github.com/emccode/libstorage/api/types"
)

// ParseTLSConfig returns a new TLS configuration.
func ParseTLSConfig(
	config gofig.Config,
	fields log.Fields,
	roots ...string) (*tls.Config, error) {

	f := func(k string, v interface{}) {
		if fields == nil {
			return
		}
		fields[k] = v
	}

	if !isSet(config, types.ConfigTLS, roots...) {
		return nil, nil
	}

	if isSet(config, types.ConfigTLSDisabled, roots...) {
		tlsDisabled := getBool(config, types.ConfigTLSDisabled, roots...)
		if tlsDisabled {
			f(types.ConfigTLSDisabled, true)
			return nil, nil
		}
	}

	if !isSet(config, types.ConfigTLSKeyFile, roots...) {
		return nil, goof.New("keyFile required")
	}
	keyFile := getString(config, types.ConfigTLSKeyFile, roots...)
	if !gotil.FileExists(keyFile) {
		return nil, goof.WithField("path", keyFile, "invalid key file")
	}
	f(types.ConfigTLSKeyFile, keyFile)

	if !isSet(config, types.ConfigTLSCertFile, roots...) {
		return nil, goof.New("certFile required")
	}
	certFile := getString(config, types.ConfigTLSCertFile, roots...)
	if !gotil.FileExists(certFile) {
		return nil, goof.WithField("path", certFile, "invalid cert file")
	}
	f(types.ConfigTLSCertFile, certFile)

	cer, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{Certificates: []tls.Certificate{cer}}

	if isSet(config, types.ConfigTLSServerName, roots...) {
		serverName := getString(config, types.ConfigTLSServerName, roots...)
		tlsConfig.ServerName = serverName
		f(types.ConfigTLSServerName, serverName)
	}

	if isSet(config, types.ConfigTLSClientCertRequired, roots...) {
		clientCertRequired := getBool(
			config, types.ConfigTLSClientCertRequired, roots...)
		if clientCertRequired {
			tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		}
		f(types.ConfigTLSClientCertRequired, clientCertRequired)
	}

	if isSet(config, types.ConfigTLSTrustedCertsFile, roots...) {
		trustedCertsFile := getString(
			config, types.ConfigTLSTrustedCertsFile, roots...)

		if !gotil.FileExists(trustedCertsFile) {
			return nil, goof.WithField(
				"path", trustedCertsFile, "invalid trust file")
		}

		f(types.ConfigTLSTrustedCertsFile, trustedCertsFile)

		buf, err := func() ([]byte, error) {
			f, err := os.Open(trustedCertsFile)
			if err != nil {
				return nil, err
			}
			defer f.Close()
			buf, err := ioutil.ReadAll(f)
			if err != nil {
				return nil, err
			}
			return buf, nil
		}()
		if err != nil {
			return nil, err
		}

		certPool := x509.NewCertPool()
		certPool.AppendCertsFromPEM(buf)
		tlsConfig.RootCAs = certPool
		tlsConfig.ClientCAs = certPool
	}

	return tlsConfig, nil
}
