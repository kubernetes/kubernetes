package client

import (
	"io/ioutil"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"net/http"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/network"
)

// GetKubeConfigOrInClusterConfig loads in-cluster config if kubeConfigFile is empty or the file if not,
// then applies overrides.
func GetKubeConfigOrInClusterConfig(kubeConfigFile string, overrides *ClientConnectionOverrides) (*rest.Config, error) {
	if len(kubeConfigFile) > 0 {
		return GetClientConfig(kubeConfigFile, overrides)
	}

	clientConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}

	applyClientConnectionOverrides(overrides, clientConfig)

	t := ClientTransportOverrides{WrapTransport: clientConfig.WrapTransport}
	if overrides != nil {
		t.MaxIdleConnsPerHost = overrides.MaxIdleConnsPerHost
	}
	clientConfig.WrapTransport = t.DefaultClientTransport

	return clientConfig, nil
}

// GetClientConfig returns the rest.Config for a kubeconfig file
func GetClientConfig(kubeConfigFile string, overrides *ClientConnectionOverrides) (*rest.Config, error) {
	kubeConfigBytes, err := ioutil.ReadFile(kubeConfigFile)
	if err != nil {
		return nil, err
	}
	kubeConfig, err := clientcmd.NewClientConfigFromBytes(kubeConfigBytes)
	if err != nil {
		return nil, err
	}
	clientConfig, err := kubeConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	applyClientConnectionOverrides(overrides, clientConfig)

	t := ClientTransportOverrides{WrapTransport: clientConfig.WrapTransport}
	if overrides != nil {
		t.MaxIdleConnsPerHost = overrides.MaxIdleConnsPerHost
	}
	clientConfig.WrapTransport = t.DefaultClientTransport

	return clientConfig, nil
}

// applyClientConnectionOverrides updates a kubeConfig with the overrides from the config.
func applyClientConnectionOverrides(overrides *ClientConnectionOverrides, kubeConfig *rest.Config) {
	if overrides == nil {
		return
	}
	if overrides.QPS > 0 {
		kubeConfig.QPS = overrides.QPS
	}
	if overrides.Burst > 0 {
		kubeConfig.Burst = int(overrides.Burst)
	}
	if len(overrides.AcceptContentTypes) > 0 {
		kubeConfig.ContentConfig.AcceptContentTypes = overrides.AcceptContentTypes
	}
	if len(overrides.ContentType) > 0 {
		kubeConfig.ContentConfig.ContentType = overrides.ContentType
	}

	// TODO both of these default values look wrong
	// if we have no preferences at this point, claim that we accept both proto and json.  We will get proto if the server supports it.
	// this is a slightly niggly thing.  If the server has proto and our client does not (possible, but not super likely) then this fails.
	if len(kubeConfig.ContentConfig.AcceptContentTypes) == 0 {
		kubeConfig.ContentConfig.AcceptContentTypes = "application/vnd.kubernetes.protobuf,application/json"
	}
	if len(kubeConfig.ContentConfig.ContentType) == 0 {
		kubeConfig.ContentConfig.ContentType = "application/vnd.kubernetes.protobuf"
	}
}

type ClientTransportOverrides struct {
	WrapTransport       func(rt http.RoundTripper) http.RoundTripper
	MaxIdleConnsPerHost int
}

// defaultClientTransport sets defaults for a client Transport that are suitable for use by infrastructure components.
func (c ClientTransportOverrides) DefaultClientTransport(rt http.RoundTripper) http.RoundTripper {
	transport, ok := rt.(*http.Transport)
	if !ok {
		return rt
	}

	transport.DialContext = network.DefaultClientDialContext()

	// Hold open more internal idle connections
	transport.MaxIdleConnsPerHost = 100
	if c.MaxIdleConnsPerHost > 0 {
		transport.MaxIdleConnsPerHost = c.MaxIdleConnsPerHost
	}

	if c.WrapTransport == nil {
		return transport

	}
	return c.WrapTransport(transport)
}

// ClientConnectionOverrides allows overriding values for rest.Config not held in a kubeconfig.  Most commonly used
// for QPS.  Empty values are not used.
type ClientConnectionOverrides struct {
	configv1.ClientConnectionOverrides

	// MaxIdleConnsPerHost, if non-zero, controls the maximum idle (keep-alive) connections to keep per-host:port.
	// If zero, DefaultMaxIdleConnsPerHost is used.
	// TODO roll this into the connection overrides in api
	MaxIdleConnsPerHost int
}
