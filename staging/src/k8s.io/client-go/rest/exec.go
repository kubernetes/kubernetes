/*
Copyright 2020 The Kubernetes Authors.

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

package rest

import (
	"fmt"
	"net/http"
	"net/url"

	clientauthenticationapi "k8s.io/client-go/pkg/apis/clientauthentication"
)

// This file contains Config logic related to exec credential plugins.

// ConfigToExecCluster creates a clientauthenticationapi.Cluster with the corresponding fields from
// the provided Config.
func ConfigToExecCluster(config *Config) (*clientauthenticationapi.Cluster, error) {
	caData, err := dataFromSliceOrFile(config.CAData, config.CAFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load CA bundle for execProvider: %v", err)
	}

	var proxyURL string
	if config.Proxy != nil {
		req, err := http.NewRequest("", config.Host, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create proxy URL request for execProvider: %w", err)
		}
		url, err := config.Proxy(req)
		if err != nil {
			return nil, fmt.Errorf("failed to get proxy URL for execProvider: %w", err)
		}
		if url != nil {
			proxyURL = url.String()
		}
	}

	return &clientauthenticationapi.Cluster{
		Server:                   config.Host,
		TLSServerName:            config.ServerName,
		InsecureSkipTLSVerify:    config.Insecure,
		CertificateAuthorityData: caData,
		ProxyURL:                 proxyURL,
		Config:                   config.ExecProvider.Config,
	}, nil
}

// ExecClusterToConfig creates a Config with the corresponding fields from the provided
// clientauthenticationapi.Cluster. The returned Config will be anonymous (i.e., it will not have
// any authentication-related fields set).
func ExecClusterToConfig(cluster *clientauthenticationapi.Cluster) (*Config, error) {
	var proxy func(*http.Request) (*url.URL, error)
	if cluster.ProxyURL != "" {
		proxyURL, err := url.Parse(cluster.ProxyURL)
		if err != nil {
			return nil, fmt.Errorf("cannot parse proxy URL: %w", err)
		}
		proxy = http.ProxyURL(proxyURL)
	}

	return &Config{
		Host: cluster.Server,
		TLSClientConfig: TLSClientConfig{
			Insecure:   cluster.InsecureSkipTLSVerify,
			ServerName: cluster.TLSServerName,
			CAData:     cluster.CertificateAuthorityData,
		},
		Proxy: proxy,
	}, nil
}
