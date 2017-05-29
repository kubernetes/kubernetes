// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package clientv3

import (
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"time"

	"github.com/coreos/etcd/pkg/tlsutil"
	"github.com/ghodss/yaml"
)

type Config struct {
	// Endpoints is a list of URLs
	Endpoints []string

	// AutoSyncInterval is the interval to update endpoints with its latest members.
	// 0 disables auto-sync. By default auto-sync is disabled.
	AutoSyncInterval time.Duration

	// DialTimeout is the timeout for failing to establish a connection.
	DialTimeout time.Duration

	// TLS holds the client secure credentials, if any.
	TLS *tls.Config

	// Username is a username for authentication
	Username string

	// Password is a password for authentication
	Password string
}

type yamlConfig struct {
	Endpoints             []string      `json:"endpoints"`
	AutoSyncInterval      time.Duration `json:"auto-sync-interval"`
	DialTimeout           time.Duration `json:"dial-timeout"`
	InsecureTransport     bool          `json:"insecure-transport"`
	InsecureSkipTLSVerify bool          `json:"insecure-skip-tls-verify"`
	Certfile              string        `json:"cert-file"`
	Keyfile               string        `json:"key-file"`
	CAfile                string        `json:"ca-file"`
}

func configFromFile(fpath string) (*Config, error) {
	b, err := ioutil.ReadFile(fpath)
	if err != nil {
		return nil, err
	}

	yc := &yamlConfig{}

	err = yaml.Unmarshal(b, yc)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		Endpoints:        yc.Endpoints,
		AutoSyncInterval: yc.AutoSyncInterval,
		DialTimeout:      yc.DialTimeout,
	}

	if yc.InsecureTransport {
		cfg.TLS = nil
		return cfg, nil
	}

	var (
		cert *tls.Certificate
		cp   *x509.CertPool
	)

	if yc.Certfile != "" && yc.Keyfile != "" {
		cert, err = tlsutil.NewCert(yc.Certfile, yc.Keyfile, nil)
		if err != nil {
			return nil, err
		}
	}

	if yc.CAfile != "" {
		cp, err = tlsutil.NewCertPool([]string{yc.CAfile})
		if err != nil {
			return nil, err
		}
	}

	tlscfg := &tls.Config{
		MinVersion:         tls.VersionTLS10,
		InsecureSkipVerify: yc.InsecureSkipTLSVerify,
		RootCAs:            cp,
	}
	if cert != nil {
		tlscfg.Certificates = []tls.Certificate{*cert}
	}
	cfg.TLS = tlscfg

	return cfg, nil
}
