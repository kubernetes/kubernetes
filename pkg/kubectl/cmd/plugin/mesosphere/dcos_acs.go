/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package mesosphere

import (
	"net/http"
	"os"

	"github.com/golang/glog"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	cmdplugin "k8s.io/kubernetes/pkg/kubectl/cmd/plugin"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const PluginACS = "mesosphere/acs"

func RegisterPlugins() error {
	return cmdplugin.Register(&acsPlugin{})
}

type acsPlugin struct {
	*cmdutil.EmptyConfigOption
}

func (p *acsPlugin) Name() string {
	return PluginACS
}

func (p *acsPlugin) ConfigOption() cmdutil.ConfigOption {
	return p
}

func (p *acsPlugin) ClientConfig(c clientcmd.ClientConfig) clientcmd.ClientConfig {
	const DCOS_TOKEN_ENV = "DCOS_TOKEN"
	return &wrappedClientConfig{
		delegate:  c,
		dcosToken: os.Getenv(DCOS_TOKEN_ENV),
	}
}

// roundTripperFunc implements http.RoundTripper
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type wrappedClientConfig struct {
	delegate  clientcmd.ClientConfig
	dcosToken string
}

// Namespace returns the namespace resulting from the merged
// result of all overrides and a boolean indicating if it was
// overridden
func (w *wrappedClientConfig) Namespace() (string, bool, error) {
	return w.delegate.Namespace()
}

// RawConfig returns the merged result of all overrides
func (w *wrappedClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return w.delegate.RawConfig()
}

// ClientConfig returns a complete client config
func (w *wrappedClientConfig) ClientConfig() (*client.Config, error) {
	config, err := w.delegate.ClientConfig()
	if w.dcosToken == "" || err != nil {
		return config, err
	}

	config.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			glog.V(6).Infoln("Using " + PluginACS + " authentication")
			// don't modify the header map of the request directly, or the request
			// for that matter; clone it then modify it.
			h := make(http.Header, len(req.Header)+1)
			for k, v := range req.Header {
				h[k] = v
			}
			h.Set("Authorization", "token="+w.dcosToken)

			clonedReq := *req
			clonedReq.Header = h
			return rt.RoundTrip(&clonedReq)
		})
	}
	return config, nil
}
