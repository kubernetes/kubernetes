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
	"net"
	"net/url"
	"strconv"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// GetControlPlaneEndpoint returns a properly formatted endpoint for the control plane built according following rules:
// - If the controlPlaneEndpoint is defined, use it.
// - if the controlPlaneEndpoint is defined but without a port number, use the controlPlaneEndpoint + localEndpoint.BindPort is used.
// - Otherwise, in case the controlPlaneEndpoint is not defined, use the localEndpoint.AdvertiseAddress + the localEndpoint.BindPort.
func GetControlPlaneEndpoint(controlPlaneEndpoint string, localEndpoint *kubeadmapi.APIEndpoint) (string, error) {
	// get the URL of the local endpoint
	localAPIEndpoint, err := GetLocalAPIEndpoint(localEndpoint)
	if err != nil {
		return "", err
	}

	// if the controlplane endpoint is defined
	if len(controlPlaneEndpoint) > 0 {
		// parse the controlplane endpoint
		var host, port string
		var err error
		if host, port, err = ParseHostPort(controlPlaneEndpoint); err != nil {
			return "", errors.Wrapf(err, "invalid value %q given for controlPlaneEndpoint", controlPlaneEndpoint)
		}

		// if a port is provided within the controlPlaneAddress warn the users we are using it, else use the bindport
		localEndpointPort := strconv.Itoa(int(localEndpoint.BindPort))
		if port != "" {
			if port != localEndpointPort {
				klog.Warning("[endpoint] WARNING: port specified in controlPlaneEndpoint overrides bindPort in the controlplane address")
			}
		} else {
			port = localEndpointPort
		}

		// overrides the control-plane url using the controlPlaneAddress (and eventually the bindport)
		return formatURL(host, port).String(), nil
	}

	return localAPIEndpoint, nil
}

// GetLocalAPIEndpoint parses an APIEndpoint and returns it as a string,
// or returns and error in case it cannot be parsed.
func GetLocalAPIEndpoint(localEndpoint *kubeadmapi.APIEndpoint) (string, error) {
	// get the URL of the local endpoint
	localEndpointIP, localEndpointPort, err := parseAPIEndpoint(localEndpoint)
	if err != nil {
		return "", err
	}
	url := formatURL(localEndpointIP.String(), localEndpointPort)
	return url.String(), nil
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
			return "", "", errors.Errorf("hostport %s: port %s must be a valid number between 1 and 65535, inclusive", hostport, port)
		}
	}

	// if host is a valid IP, returns it
	if ip := netutils.ParseIPSloppy(host); ip != nil {
		return host, port, nil
	}

	// if host is a validate RFC-1123 subdomain, returns it
	if errs := validation.IsDNS1123Subdomain(host); len(errs) == 0 {
		return host, port, nil
	}

	return "", "", errors.Errorf("hostport %s: host '%s' must be a valid IP address or a valid RFC-1123 DNS subdomain", hostport, host)
}

// ParsePort parses a string representing a TCP port.
// If the string is not a valid representation of a TCP port, ParsePort returns an error.
func ParsePort(port string) (int, error) {
	portInt, err := netutils.ParsePort(port, true)
	if err == nil && (1 <= portInt && portInt <= 65535) {
		return portInt, nil
	}

	return 0, errors.New("port must be a valid number between 1 and 65535, inclusive")
}

// parseAPIEndpoint parses an APIEndpoint and returns the AdvertiseAddress as net.IP and the BindPort as string.
// If the BindPort or AdvertiseAddress are invalid it returns an error.
func parseAPIEndpoint(localEndpoint *kubeadmapi.APIEndpoint) (net.IP, string, error) {
	// parse the bind port
	bindPortString := strconv.Itoa(int(localEndpoint.BindPort))
	if _, err := ParsePort(bindPortString); err != nil {
		return nil, "", errors.Wrapf(err, "invalid value %q given for api.bindPort", localEndpoint.BindPort)
	}

	// parse the AdvertiseAddress
	var ip = netutils.ParseIPSloppy(localEndpoint.AdvertiseAddress)
	if ip == nil {
		return nil, "", errors.Errorf("invalid value `%s` given for api.advertiseAddress", localEndpoint.AdvertiseAddress)
	}

	return ip, bindPortString, nil
}

// formatURL takes a host and a port string and creates a net.URL using https scheme
func formatURL(host, port string) *url.URL {
	return &url.URL{
		Scheme: "https",
		Host:   net.JoinHostPort(host, port),
	}
}
