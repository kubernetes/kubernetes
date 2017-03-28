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

// Package options contains flags and options for initializing an apiserver
package options

import (
	"fmt"
	"net"
	"strconv"

	"github.com/pborman/uuid"
	"github.com/spf13/pflag"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	kubeserver "k8s.io/kubernetes/pkg/kubeapiserver/server"
)

// NewSecureServingOptions gives default values for the kube-apiserver and federation-apiserver which are not the options wanted by
// "normal" API servers running on the platform
func NewSecureServingOptions() *genericoptions.SecureServingOptions {
	return &genericoptions.SecureServingOptions{
		BindAddress: net.ParseIP("0.0.0.0"),
		BindPort:    6443,
		ServerCert: genericoptions.GeneratableKeyCert{
			PairName:      "apiserver",
			CertDirectory: "/var/run/kubernetes",
		},
	}
}

// DefaultAdvertiseAddress sets the field AdvertiseAddress if
// unset. The field will be set based on the SecureServingOptions. If
// the SecureServingOptions is not present, DefaultExternalAddress
// will fall back to the insecure ServingOptions.
func DefaultAdvertiseAddress(s *genericoptions.ServerRunOptions, insecure *InsecureServingOptions) error {
	if insecure == nil {
		return nil
	}

	if s.AdvertiseAddress == nil || s.AdvertiseAddress.IsUnspecified() {
		hostIP, err := insecure.DefaultExternalAddress()
		if err != nil {
			return fmt.Errorf("Unable to find suitable network address.error='%v'. "+
				"Try to set the AdvertiseAddress directly or provide a valid BindAddress to fix this.", err)
		}
		s.AdvertiseAddress = hostIP
	}

	return nil
}

// InsecureServingOptions are for creating an unauthenticated, unauthorized, insecure port.
// No one should be using these anymore.
type InsecureServingOptions struct {
	BindAddress net.IP
	BindPort    int
}

// NewInsecureServingOptions is for creating an unauthenticated, unauthorized, insecure port.
// No one should be using these anymore.
func NewInsecureServingOptions() *InsecureServingOptions {
	return &InsecureServingOptions{
		BindAddress: net.ParseIP("127.0.0.1"),
		BindPort:    8080,
	}
}

func (s InsecureServingOptions) Validate(portArg string) []error {
	errors := []error{}

	if s.BindPort < 0 || s.BindPort > 65535 {
		errors = append(errors, fmt.Errorf("--insecure-port %v must be between 0 and 65535, inclusive. 0 for turning off secure port.", s.BindPort))
	}

	return errors
}

func (s *InsecureServingOptions) DefaultExternalAddress() (net.IP, error) {
	return utilnet.ChooseBindAddress(s.BindAddress)
}

func (s *InsecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "insecure-bind-address", s.BindAddress, ""+
		"The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). "+
		"Defaults to localhost.")

	fs.IntVar(&s.BindPort, "insecure-port", s.BindPort, ""+
		"The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed "+
		"that firewall rules are set up such that this port is not reachable from outside of "+
		"the cluster and that port 443 on the cluster's public address is proxied to this "+
		"port. This is performed by nginx in the default setup.")
}

func (s *InsecureServingOptions) AddDeprecatedFlags(fs *pflag.FlagSet) {
	fs.IPVar(&s.BindAddress, "address", s.BindAddress,
		"DEPRECATED: see --insecure-bind-address instead.")
	fs.MarkDeprecated("address", "see --insecure-bind-address instead.")

	fs.IntVar(&s.BindPort, "port", s.BindPort, "DEPRECATED: see --insecure-port instead.")
	fs.MarkDeprecated("port", "see --insecure-port instead.")
}

func (s *InsecureServingOptions) ApplyTo(c *server.Config) (*kubeserver.InsecureServingInfo, error) {
	if s.BindPort <= 0 {
		return nil, nil
	}

	ret := &kubeserver.InsecureServingInfo{
		BindAddress: net.JoinHostPort(s.BindAddress.String(), strconv.Itoa(s.BindPort)),
	}

	var err error
	privilegedLoopbackToken := uuid.NewRandom().String()
	if c.LoopbackClientConfig, err = ret.NewLoopbackClientConfig(privilegedLoopbackToken); err != nil {
		return nil, err
	}

	return ret, nil
}
