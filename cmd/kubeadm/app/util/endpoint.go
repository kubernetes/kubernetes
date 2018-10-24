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

package util

import (
	"fmt"
	"net"
	"net/url"
	"strconv"

	"k8s.io/apimachinery/pkg/util/validation"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// GetMasterEndpoint returns a properly formatted endpoint for the control plane built according following rules:
// - If the ControlPlaneEndpoint is defined, use it.
// - if the ControlPlaneEndpoint is defined but without a port number, use the ControlPlaneEndpoint + api.BindPort is used.
// - Otherwise, in case the ControlPlaneEndpoint is not defined, use the api.AdvertiseAddress + the api.BindPort.
func GetMasterEndpoint(cfg *kubeadmapi.InitConfiguration) (string, error) {
	// parse the bind port
	bindPortString := strconv.Itoa(int(cfg.APIEndpoint.BindPort))
	if _, err := ParsePort(bindPortString); err != nil {
		return "", fmt.Errorf("invalid value %q given for api.bindPort: %s", cfg.APIEndpoint.BindPort, err)
	}

	// parse the AdvertiseAddress
	var ip = net.ParseIP(cfg.APIEndpoint.AdvertiseAddress)
	if ip == nil {
		return "", fmt.Errorf("invalid value `%s` given for api.advertiseAddress", cfg.APIEndpoint.AdvertiseAddress)
	}

	// set the master url using cfg.API.AdvertiseAddress + the cfg.API.BindPort
	masterURL := &url.URL{
		Scheme: "https",
		Host:   net.JoinHostPort(ip.String(), bindPortString),
	}

	// if the controlplane endpoint is defined
	if len(cfg.ControlPlaneEndpoint) > 0 {
		// parse the controlplane endpoint
		var host, port string
		var err error
		if host, port, err = ParseHostPort(cfg.ControlPlaneEndpoint); err != nil {
			return "", fmt.Errorf("invalid value %q given for controlPlaneEndpoint: %s", cfg.ControlPlaneEndpoint, err)
		}

		// if a port is provided within the controlPlaneAddress warn the users we are using it, else use the bindport
		if port != "" {
			if port != bindPortString {
				fmt.Println("[endpoint] WARNING: port specified in controlPlaneEndpoint overrides bindPort in the controlplane address")
			}
		} else {
			port = bindPortString
		}

		// overrides the master url using the controlPlaneAddress (and eventually the bindport)
		masterURL = &url.URL{
			Scheme: "https",
			Host:   net.JoinHostPort(host, port),
		}
	}

	return masterURL.String(), nil
}

// ParseHostPort parses a network address of the form "host:port", "ipv4:port", "[ipv6]:port" into host and port;
// ":port" can be eventually omitted.
// If the string is not a valid representation of network address, ParseHostPort returns an error.
func ParseHostPort(hostport string) (string, string, error) {
	var host, port string
	var err error

	// try to split host and port
	if host, port, err = net.SplitHostPort(hostport); err != nil {
		// if SplitHostPort returns an error, the entire hostport is considered as host
		host = hostport
	}

	// if port is defined, parse and validate it
	if port != "" {
		if _, err := ParsePort(port); err != nil {
			return "", "", fmt.Errorf("port must be a valid number between 1 and 65535, inclusive")
		}
	}

	// if host is a valid IP, returns it
	if ip := net.ParseIP(host); ip != nil {
		return host, port, nil
	}

	// if host is a validate RFC-1123 subdomain, returns it
	if errs := validation.IsDNS1123Subdomain(host); len(errs) == 0 {
		return host, port, nil
	}

	return "", "", fmt.Errorf("host must be a valid IP address or a valid RFC-1123 DNS subdomain")
}

// ParsePort parses a string representing a TCP port.
// If the string is not a valid representation of a TCP port, ParsePort returns an error.
func ParsePort(port string) (int, error) {
	portInt, err := strconv.Atoi(port)
	if err == nil && (1 <= portInt && portInt <= 65535) {
		return portInt, nil
	}

	return 0, fmt.Errorf("port must be a valid number between 1 and 65535, inclusive")
}
