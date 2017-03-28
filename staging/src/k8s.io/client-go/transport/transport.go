/*
Copyright 2015 The Kubernetes Authors.

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

package transport

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
)

// New returns an http.RoundTripper that will provide the authentication
// or transport level security defined by the provided Config.
func New(config *Config) (http.RoundTripper, error) {
	// Set transport level security
	if config.Transport != nil && (config.HasCA() || config.HasCertAuth() || config.TLS.Insecure) {
		return nil, fmt.Errorf("using a custom transport with TLS certificate options or the insecure flag is not allowed")
	}

	var (
		rt  http.RoundTripper
		err error
	)

	if config.Transport != nil {
		rt = config.Transport
	} else {
		rt, err = tlsCache.get(config)
		if err != nil {
			return nil, err
		}
	}

	return HTTPWrappersForConfig(config, rt)
}

// TLSConfigFor returns a tls.Config that will provide the transport level security defined
// by the provided Config. Will return nil if no transport level security is requested.
func TLSConfigFor(c *Config) (*tls.Config, error) {
	if !(c.HasCA() || c.HasCertAuth() || c.TLS.Insecure) {
		return nil, nil
	}
	if c.HasCA() && c.TLS.Insecure {
		return nil, fmt.Errorf("specifying a root certificates file with the insecure flag is not allowed")
	}
	if err := loadTLSFiles(c); err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{
		// Can't use SSLv3 because of POODLE and BEAST
		// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
		// Can't use TLSv1.1 because of RC4 cipher usage
		MinVersion:         tls.VersionTLS12,
		InsecureSkipVerify: c.TLS.Insecure,
		ServerName:         c.TLS.ServerName,
	}

	if c.HasCA() {
		tlsConfig.RootCAs = rootCertPool(c.TLS.CAData)
	}

	if c.HasCertAuth() {
		cert, err := tls.X509KeyPair(c.TLS.CertData, c.TLS.KeyData)
		if err != nil {
			return nil, err
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	return tlsConfig, nil
}

// loadTLSFiles copies the data from the CertFile, KeyFile, and CAFile fields into the CertData,
// KeyData, and CAFile fields, or returns an error. If no error is returned, all three fields are
// either populated or were empty to start.
func loadTLSFiles(c *Config) error {
	var err error
	c.TLS.CAData, err = dataFromSliceOrFile(c.TLS.CAData, c.TLS.CAFile)
	if err != nil {
		return err
	}

	c.TLS.CertData, err = dataFromSliceOrFile(c.TLS.CertData, c.TLS.CertFile)
	if err != nil {
		return err
	}

	c.TLS.KeyData, err = dataFromSliceOrFile(c.TLS.KeyData, c.TLS.KeyFile)
	if err != nil {
		return err
	}
	return nil
}

// dataFromSliceOrFile returns data from the slice (if non-empty), or from the file,
// or an error if an error occurred reading the file
func dataFromSliceOrFile(data []byte, file string) ([]byte, error) {
	if len(data) > 0 {
		return data, nil
	}
	if len(file) > 0 {
		fileData, err := ioutil.ReadFile(file)
		if err != nil {
			return []byte{}, err
		}
		return fileData, nil
	}
	return nil, nil
}

// rootCertPool returns nil if caData is empty.  When passed along, this will mean "use system CAs".
// When caData is not empty, it will be the ONLY information used in the CertPool.
func rootCertPool(caData []byte) *x509.CertPool {
	// What we really want is a copy of x509.systemRootsPool, but that isn't exposed.  It's difficult to build (see the go
	// code for a look at the platform specific insanity), so we'll use the fact that RootCAs == nil gives us the system values
	// It doesn't allow trusting either/or, but hopefully that won't be an issue
	if len(caData) == 0 {
		return nil
	}

	// if we have caData, use it
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caData)
	return certPool
}
