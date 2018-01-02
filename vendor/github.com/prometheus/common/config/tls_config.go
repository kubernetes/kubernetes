// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package config

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
)

// TLSConfig configures the options for TLS connections.
type TLSConfig struct {
	// The CA cert to use for the targets.
	CAFile string `yaml:"ca_file,omitempty"`
	// The client cert file for the targets.
	CertFile string `yaml:"cert_file,omitempty"`
	// The client key file for the targets.
	KeyFile string `yaml:"key_file,omitempty"`
	// Disable target certificate validation.
	InsecureSkipVerify bool `yaml:"insecure_skip_verify"`

	// Catches all undefined fields and must be empty after parsing.
	XXX map[string]interface{} `yaml:",inline"`
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (c *TLSConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type plain TLSConfig
	if err := unmarshal((*plain)(c)); err != nil {
		return err
	}
	return checkOverflow(c.XXX, "TLS config")
}

// GenerateConfig produces a tls.Config based on TLS connection options.
// It loads certificate files from disk if they are defined.
func (c *TLSConfig) GenerateConfig() (*tls.Config, error) {
	tlsConfig := &tls.Config{InsecureSkipVerify: c.InsecureSkipVerify}

	// If a CA cert is provided then let's read it in so we can validate the
	// scrape target's certificate properly.
	if len(c.CAFile) > 0 {
		caCertPool := x509.NewCertPool()
		// Load CA cert.
		caCert, err := ioutil.ReadFile(c.CAFile)
		if err != nil {
			return nil, fmt.Errorf("unable to use specified CA cert %s: %s", c.CAFile, err)
		}
		caCertPool.AppendCertsFromPEM(caCert)
		tlsConfig.RootCAs = caCertPool
	}

	if len(c.CertFile) > 0 && len(c.KeyFile) == 0 {
		return nil, fmt.Errorf("client cert file %q specified without client key file", c.CertFile)
	} else if len(c.KeyFile) > 0 && len(c.CertFile) == 0 {
		return nil, fmt.Errorf("client key file %q specified without client cert file", c.KeyFile)
	} else if len(c.CertFile) > 0 && len(c.KeyFile) > 0 {
		cert, err := tls.LoadX509KeyPair(c.CertFile, c.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("unable to use specified client cert (%s) & key (%s): %s", c.CertFile, c.KeyFile, err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}
	tlsConfig.BuildNameToCertificate()

	return tlsConfig, nil
}
