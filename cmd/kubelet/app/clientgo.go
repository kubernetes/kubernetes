/*
Copyright 2017 The Kubernetes Authors.

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

// this file contains parallel methods for getting a client-go clientconfig
// The types are not reasonably compatibly between the client-go and kubernetes restclient.Interface,
// restclient.Config, or typed clients, so this is the simplest solution during the transition
package app

import (
	"fmt"
	"net/http"

	"github.com/golang/glog"

	"k8s.io/client-go/rest"
	clientauth "k8s.io/client-go/tools/auth"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/client/chaosclient"
)

func kubeconfigClientGoConfig(s *options.KubeletServer) (*rest.Config, error) {
	if s.RequireKubeConfig {
		// Ignores the values of s.APIServerList
		return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig.Value()},
			&clientcmd.ConfigOverrides{},
		).ClientConfig()
	}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.KubeConfig.Value()},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.APIServerList[0]}},
	).ClientConfig()
}

// createClientConfig creates a client configuration from the command line
// arguments. If --kubeconfig is explicitly set, it will be used. If it is
// not set, we attempt to load the default kubeconfig file, and if we cannot,
// we fall back to the default client with no auth - this fallback does not, in
// and of itself, constitute an error.
func createClientGoConfig(s *options.KubeletServer) (*rest.Config, error) {
	if s.RequireKubeConfig {
		return kubeconfigClientGoConfig(s)
	}

	// TODO: handle a new --standalone flag that bypasses kubeconfig loading and returns no error.
	// DEPRECATED: all subsequent code is deprecated
	if len(s.APIServerList) == 0 {
		return nil, fmt.Errorf("no api servers specified")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(s.APIServerList) > 1 {
		glog.Infof("Multiple api servers specified.  Picking first one")
	}

	if s.KubeConfig.Provided() {
		return kubeconfigClientGoConfig(s)
	}
	// If KubeConfig was not provided, try to load the default file, then fall back
	// to a default auth config.
	clientConfig, err := kubeconfigClientGoConfig(s)
	if err != nil {
		glog.Warningf("Could not load kubeconfig file %s: %v. Using default client config instead.", s.KubeConfig, err)

		authInfo := &clientauth.Info{}
		authConfig, err := authInfo.MergeWithConfig(rest.Config{})
		if err != nil {
			return nil, err
		}
		authConfig.Host = s.APIServerList[0]
		clientConfig = &authConfig
	}
	return clientConfig, nil
}

// createAPIServerClientGoConfig generates a client.Config from command line flags,
// including api-server-list, via createClientConfig and then injects chaos into
// the configuration via addChaosToClientConfig. This func is exported to support
// integration with third party kubelet extensions (e.g. kubernetes-mesos).
func createAPIServerClientGoConfig(s *options.KubeletServer) (*rest.Config, error) {
	clientConfig, err := createClientGoConfig(s)
	if err != nil {
		return nil, err
	}

	clientConfig.ContentType = s.ContentType
	// Override kubeconfig qps/burst settings from flags
	clientConfig.QPS = float32(s.KubeAPIQPS)
	clientConfig.Burst = int(s.KubeAPIBurst)

	addChaosToClientGoConfig(s, clientConfig)
	return clientConfig, nil
}

// addChaosToClientConfig injects random errors into client connections if configured.
func addChaosToClientGoConfig(s *options.KubeletServer, config *rest.Config) {
	if s.ChaosChance != 0.0 {
		config.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
			seed := chaosclient.NewSeed(1)
			// TODO: introduce a standard chaos package with more tunables - this is just a proof of concept
			// TODO: introduce random latency and stalls
			return chaosclient.NewChaosRoundTripper(rt, chaosclient.LogChaos, seed.P(s.ChaosChance, chaosclient.ErrSimulatedConnectionResetByPeer))
		}
	}
}
