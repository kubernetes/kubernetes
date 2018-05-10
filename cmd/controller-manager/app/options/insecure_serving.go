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
	"k8s.io/apiserver/pkg/server/options"
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
)

// InsecureServingOptions are for creating an unauthenticated, unauthorized, insecure port.
// No one should be using these anymore.
type InsecureServingOptions struct {
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
	ListenFunc func(network, addr string) (net.Listener, int, error)
}

// Validate ensures that the insecure port values within the range of the port.
func (s *InsecureServingOptions) Validate() []error {
	if s == nil {
		return nil
	}

	errors := []error{}

	if s.BindPort < 0 || s.BindPort > 32767 {
		errors = append(errors, fmt.Errorf("--insecure-port %v must be between 0 and 32767, inclusive. 0 for turning off insecure (HTTP) port", s.BindPort))
	}

	return errors
}

// AddFlags adds flags related to insecure serving for controller manager to the specified FlagSet.
func (s *InsecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	if s == nil {
		return
	}

	fs.IPVar(&s.BindAddress, "address", s.BindAddress, "DEPRECATED: the IP address on which to listen for the --port port (set to 0.0.0.0 for all IPv4 interfaces and :: for all IPv6 interfaces). See --bind-address instead.")
	// MarkDeprecated hides the flag from the help. We don't want that:
	// fs.MarkDeprecated("address", "see --bind-address instead.")
	fs.IntVar(&s.BindPort, "port", s.BindPort, "DEPRECATED: the port on which to serve HTTP insecurely without authentication and authorization. If 0, don't serve HTTPS at all. See --secure-port instead.")
	// MarkDeprecated hides the flag from the help. We don't want that:
	// fs.MarkDeprecated("port", "see --secure-port instead.")
}

// ApplyTo adds InsecureServingOptions to the insecureserverinfo amd kube-controller manager configuration.
// Note: the double pointer allows to set the *InsecureServingInfo to nil without referencing the struct hosting this pointer.
func (s *InsecureServingOptions) ApplyTo(c **genericcontrollermanager.InsecureServingInfo) error {
	if s == nil {
		return nil
	}
	if s.BindPort <= 0 {
		return nil
	}

	if s.Listener == nil {
		var err error
		listen := options.CreateListener
		if s.ListenFunc != nil {
			listen = s.ListenFunc
		}
		addr := net.JoinHostPort(s.BindAddress.String(), fmt.Sprintf("%d", s.BindPort))
		s.Listener, s.BindPort, err = listen(s.BindNetwork, addr)
		if err != nil {
			return fmt.Errorf("failed to create listener: %v", err)
		}
	}

	*c = &genericcontrollermanager.InsecureServingInfo{
		Listener: s.Listener,
	}

	return nil
}
