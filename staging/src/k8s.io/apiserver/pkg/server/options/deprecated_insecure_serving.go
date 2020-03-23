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

package options

import (
	"fmt"
	"net"

	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/rest"
)

// DeprecatedInsecureServingOptions are for creating an unauthenticated, unauthorized, insecure port.
// No one should be using these anymore.
// DEPRECATED: all insecure serving options are removed in a future version
type DeprecatedInsecureServingOptions struct {
	BindAddress net.IP
	BindPort    int
	// BindNetwork is the type of network to bind to - defaults to "tcp", accepts "tcp",
	// "tcp4", and "tcp6".
	BindNetwork string

	// Listener is the secure server network listener.
	// either Listener or BindAddress/BindPort/BindNetwork is set,
	// if Listener is set, use it and omit BindAddress/BindPort/BindNetwork.
	Listener net.Listener

	// ListenFunc can be overridden to create a custom listener, e.g. for mocking in tests.
	// It defaults to options.CreateListener.
	ListenFunc func(network, addr string, config net.ListenConfig) (net.Listener, int, error)
}

// Validate ensures that the insecure port values within the range of the port.
func (s *DeprecatedInsecureServingOptions) Validate() []error {
	if s == nil {
		return nil
	}

	errors := []error{}

	if s.BindPort < 0 || s.BindPort > 65535 {
		errors = append(errors, fmt.Errorf("insecure port %v must be between 0 and 65535, inclusive. 0 for turning off insecure (HTTP) port", s.BindPort))
	}

	return errors
}

// AddFlags adds flags related to insecure serving to the specified FlagSet.
func (s *DeprecatedInsecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	if s == nil {
		return
	}

	fs.IPVar(&s.BindAddress, "insecure-bind-address", s.BindAddress, ""+
		"The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all IPv4 interfaces and :: for all IPv6 interfaces).")
	// Though this flag is deprecated, we discovered security concerns over how to do health checks without it e.g. #43784
	fs.MarkDeprecated("insecure-bind-address", "This flag will be removed in a future version.")
	fs.Lookup("insecure-bind-address").Hidden = false

	fs.IntVar(&s.BindPort, "insecure-port", s.BindPort, ""+
		"The port on which to serve unsecured, unauthenticated access.")
	// Though this flag is deprecated, we discovered security concerns over how to do health checks without it e.g. #43784
	fs.MarkDeprecated("insecure-port", "This flag will be removed in a future version.")
	fs.Lookup("insecure-port").Hidden = false
}

// AddUnqualifiedFlags adds flags related to insecure serving without the --insecure prefix to the specified FlagSet.
func (s *DeprecatedInsecureServingOptions) AddUnqualifiedFlags(fs *pflag.FlagSet) {
	if s == nil {
		return
	}

	fs.IPVar(&s.BindAddress, "address", s.BindAddress,
		"The IP address on which to serve the insecure --port (set to 0.0.0.0 for all IPv4 interfaces and :: for all IPv6 interfaces).")
	fs.MarkDeprecated("address", "see --bind-address instead.")
	fs.Lookup("address").Hidden = false

	fs.IntVar(&s.BindPort, "port", s.BindPort, "The port on which to serve unsecured, unauthenticated access. Set to 0 to disable.")
	fs.MarkDeprecated("port", "see --secure-port instead.")
	fs.Lookup("port").Hidden = false
}

// ApplyTo adds DeprecatedInsecureServingOptions to the insecureserverinfo and kube-controller manager configuration.
// Note: the double pointer allows to set the *DeprecatedInsecureServingInfo to nil without referencing the struct hosting this pointer.
func (s *DeprecatedInsecureServingOptions) ApplyTo(c **server.DeprecatedInsecureServingInfo) error {
	if s == nil {
		return nil
	}
	if s.BindPort <= 0 {
		return nil
	}

	if s.Listener == nil {
		var err error
		listen := CreateListener
		if s.ListenFunc != nil {
			listen = s.ListenFunc
		}
		addr := net.JoinHostPort(s.BindAddress.String(), fmt.Sprintf("%d", s.BindPort))
		s.Listener, s.BindPort, err = listen(s.BindNetwork, addr, net.ListenConfig{})
		if err != nil {
			return fmt.Errorf("failed to create listener: %v", err)
		}
	}

	*c = &server.DeprecatedInsecureServingInfo{
		Listener: s.Listener,
	}

	return nil
}

// WithLoopback adds loopback functionality to the serving options.
func (o *DeprecatedInsecureServingOptions) WithLoopback() *DeprecatedInsecureServingOptionsWithLoopback {
	return &DeprecatedInsecureServingOptionsWithLoopback{o}
}

// DeprecatedInsecureServingOptionsWithLoopback adds loopback functionality to the DeprecatedInsecureServingOptions.
// DEPRECATED: all insecure serving options will be removed in a future version, however note that
// there are security concerns over how health checks can work here - see e.g. #43784
type DeprecatedInsecureServingOptionsWithLoopback struct {
	*DeprecatedInsecureServingOptions
}

// ApplyTo fills up serving information in the server configuration.
func (s *DeprecatedInsecureServingOptionsWithLoopback) ApplyTo(insecureServingInfo **server.DeprecatedInsecureServingInfo, loopbackClientConfig **rest.Config) error {
	if s == nil || s.DeprecatedInsecureServingOptions == nil || insecureServingInfo == nil {
		return nil
	}

	if err := s.DeprecatedInsecureServingOptions.ApplyTo(insecureServingInfo); err != nil {
		return err
	}

	if *insecureServingInfo == nil || loopbackClientConfig == nil {
		return nil
	}

	secureLoopbackClientConfig, err := (*insecureServingInfo).NewLoopbackClientConfig()
	switch {
	// if we failed and there's no fallback loopback client config, we need to fail
	case err != nil && *loopbackClientConfig == nil:
		return err

		// if we failed, but we already have a fallback loopback client config (usually insecure), allow it
	case err != nil && *loopbackClientConfig != nil:

	default:
		*loopbackClientConfig = secureLoopbackClientConfig
	}

	return nil
}
