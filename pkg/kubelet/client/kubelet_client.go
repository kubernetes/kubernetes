/*
Copyright 2014 The Kubernetes Authors.

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

package client

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/transport"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

type KubeletClientConfig struct {
	// Default port - used if no information about Kubelet port can be found in Node.NodeStatus.DaemonEndpoints.
	Port        uint
	EnableHttps bool

	// TLSClientConfig contains settings to enable transport layer security
	restclient.TLSClientConfig

	// Server requires Bearer authentication
	BearerToken string

	// HTTPTimeout is used by the client to timeout http requests to Kubelet.
	HTTPTimeout time.Duration

	// Dial is a custom dialer used for the client
	Dial func(net, addr string) (net.Conn, error)
}

// KubeletClient is an interface for all kubelet functionality
type KubeletClient interface {
	ConnectionInfoGetter
}

type ConnectionInfoGetter interface {
	GetConnectionInfo(ctx api.Context, nodeName string) (scheme string, port uint, transport http.RoundTripper, err error)
}

// HTTPKubeletClient is the default implementation of KubeletHealthchecker, accesses the kubelet over HTTP.
type HTTPKubeletClient struct {
	Client *http.Client
	Config *KubeletClientConfig
}

func MakeTransport(config *KubeletClientConfig) (http.RoundTripper, error) {
	tlsConfig, err := transport.TLSConfigFor(config.transportConfig())
	if err != nil {
		return nil, err
	}

	rt := http.DefaultTransport
	if config.Dial != nil || tlsConfig != nil {
		rt = utilnet.SetOldTransportDefaults(&http.Transport{
			Dial:            config.Dial,
			TLSClientConfig: tlsConfig,
		})
	}

	return transport.HTTPWrappersForConfig(config.transportConfig(), rt)
}

// TODO: this structure is questionable, it should be using client.Config and overriding defaults.
func NewStaticKubeletClient(config *KubeletClientConfig) (KubeletClient, error) {
	transport, err := MakeTransport(config)
	if err != nil {
		return nil, err
	}
	c := &http.Client{
		Transport: transport,
		Timeout:   config.HTTPTimeout,
	}
	return &HTTPKubeletClient{
		Client: c,
		Config: config,
	}, nil
}

// In default HTTPKubeletClient ctx is unused.
func (c *HTTPKubeletClient) GetConnectionInfo(ctx api.Context, nodeName string) (string, uint, http.RoundTripper, error) {
	if errs := validation.ValidateNodeName(nodeName, false); len(errs) != 0 {
		return "", 0, nil, fmt.Errorf("invalid node name: %s", strings.Join(errs, ";"))
	}
	scheme := "http"
	if c.Config.EnableHttps {
		scheme = "https"
	}
	return scheme, c.Config.Port, c.Client.Transport, nil
}

// FakeKubeletClient is a fake implementation of KubeletClient which returns an error
// when called.  It is useful to pass to the master in a test configuration with
// no kubelets.
type FakeKubeletClient struct{}

func (c FakeKubeletClient) GetConnectionInfo(ctx api.Context, nodeName string) (string, uint, http.RoundTripper, error) {
	return "", 0, nil, errors.New("Not Implemented")
}

// transportConfig converts a client config to an appropriate transport config.
func (c *KubeletClientConfig) transportConfig() *transport.Config {
	cfg := &transport.Config{
		TLS: transport.TLSConfig{
			CAFile:   c.CAFile,
			CAData:   c.CAData,
			CertFile: c.CertFile,
			CertData: c.CertData,
			KeyFile:  c.KeyFile,
			KeyData:  c.KeyData,
		},
		BearerToken: c.BearerToken,
	}
	if c.EnableHttps && !cfg.HasCA() {
		cfg.TLS.Insecure = true
	}
	return cfg
}
