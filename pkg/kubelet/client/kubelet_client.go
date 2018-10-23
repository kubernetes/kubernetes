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
	"net/http"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

type KubeletClientConfig struct {
	// Default port - used if no information about Kubelet port can be found in Node.NodeStatus.DaemonEndpoints.
	Port         uint
	ReadOnlyPort uint
	EnableHttps  bool

	// PreferredAddressTypes - used to select an address from Node.NodeStatus.Addresses
	PreferredAddressTypes []string

	// TLSClientConfig contains settings to enable transport layer security
	restclient.TLSClientConfig

	// Server requires Bearer authentication
	BearerToken string

	// HTTPTimeout is used by the client to timeout http requests to Kubelet.
	HTTPTimeout time.Duration

	// Dial is a custom dialer used for the client
	Dial utilnet.DialFunc
}

// ConnectionInfo provides the information needed to connect to a kubelet
type ConnectionInfo struct {
	Scheme    string
	Hostname  string
	Port      string
	Transport http.RoundTripper
}

// ConnectionInfoGetter provides ConnectionInfo for the kubelet running on a named node
type ConnectionInfoGetter interface {
	GetConnectionInfo(nodeName types.NodeName) (*ConnectionInfo, error)
}

func MakeTransport(config *KubeletClientConfig) (http.RoundTripper, error) {
	tlsConfig, err := transport.TLSConfigFor(config.transportConfig())
	if err != nil {
		return nil, err
	}

	rt := http.DefaultTransport
	if config.Dial != nil || tlsConfig != nil {
		rt = utilnet.SetOldTransportDefaults(&http.Transport{
			DialContext:     config.Dial,
			TLSClientConfig: tlsConfig,
		})
	}

	return transport.HTTPWrappersForConfig(config.transportConfig(), rt)
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

// NodeConnectionInfoGetter obtains connection info from the status of a Node API object
type NodeConnectionInfoGetter struct {
	// nodes is used to look up Node objects
	nodeClient corev1client.NodeInterface
	// scheme is the scheme to use to connect to all kubelets
	scheme string
	// defaultPort is the port to use if no Kubelet endpoint port is recorded in the node status
	defaultPort int
	// transport is the transport to use to send a request to all kubelets
	transport http.RoundTripper
	// preferredAddressTypes specifies the preferred order to use to find a node address
	preferredAddressTypes []v1.NodeAddressType
}

func NewNodeConnectionInfoGetter(nodeClient corev1client.NodeInterface, config KubeletClientConfig) (ConnectionInfoGetter, error) {
	scheme := "http"
	if config.EnableHttps {
		scheme = "https"
	}

	transport, err := MakeTransport(&config)
	if err != nil {
		return nil, err
	}

	types := []v1.NodeAddressType{}
	for _, t := range config.PreferredAddressTypes {
		types = append(types, v1.NodeAddressType(t))
	}

	return &NodeConnectionInfoGetter{
		nodeClient:  nodeClient,
		scheme:      scheme,
		defaultPort: int(config.Port),
		transport:   transport,

		preferredAddressTypes: types,
	}, nil
}

func (k *NodeConnectionInfoGetter) GetConnectionInfo(nodeName types.NodeName) (*ConnectionInfo, error) {
	node, err := k.nodeClient.Get(string(nodeName), metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	// Find a kubelet-reported address, using preferred address type
	host, err := nodeutil.GetPreferredNodeAddress(node, k.preferredAddressTypes)
	if err != nil {
		return nil, err
	}

	// Use the kubelet-reported port, if present
	port := int(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
	if port <= 0 {
		port = k.defaultPort
	}

	return &ConnectionInfo{
		Scheme:    k.scheme,
		Hostname:  host,
		Port:      strconv.Itoa(port),
		Transport: k.transport,
	}, nil
}
