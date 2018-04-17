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
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// GetMasterEndpoint returns a properly formatted Master Endpoint
// or passes the error from GetMasterHostPort.
func GetMasterEndpoint(api *kubeadmapi.API) (string, error) {

	hostPort, err := GetMasterHostPort(api)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("https://%s", hostPort), nil
}

// GetMasterHostPort returns a properly formatted Master hostname or IP and port pair, or error
// if the hostname or IP address can not be parsed or port is outside the valid TCP range.
func GetMasterHostPort(api *kubeadmapi.API) (string, error) {
	var masterIP string
	var portStr string
	if len(api.ControlPlaneEndpoint) > 0 {
		if strings.Contains(api.ControlPlaneEndpoint, ":") {
			var err error
			masterIP, portStr, err = net.SplitHostPort(api.ControlPlaneEndpoint)
			if err != nil {
				return "", fmt.Errorf("invalid value `%s` given for `ControlPlaneEndpoint`: %s", api.ControlPlaneEndpoint, err)
			}
		} else {
			masterIP = api.ControlPlaneEndpoint
		}
		errs := validation.IsDNS1123Subdomain(masterIP)
		if len(errs) > 0 {
			return "", fmt.Errorf("error parsing `ControlPlaneEndpoint` to valid dns subdomain with errors: %s", errs)
		}
	} else {
		ip := net.ParseIP(api.AdvertiseAddress)
		if ip == nil {
			return "", fmt.Errorf("error parsing address %s", api.AdvertiseAddress)
		}
		masterIP = ip.String()
	}

	var port int32
	if len(portStr) > 0 {
		portInt, err := strconv.Atoi(portStr)
		if err != nil {
			return "", fmt.Errorf("error parsing `ControlPlaneEndpoint` port value `%s`: %s", portStr, err.Error())
		}
		port = int32(portInt)
		fmt.Println("[endpoint] WARNING: specifying a port for `ControlPlaneEndpoint` overrides `BindPort`")
	} else {
		port = api.BindPort
	}

	if port < 0 || port > 65535 {
		return "", fmt.Errorf("api server port must be between 0 and 65535, %d was given", port)
	}

	hostPort := net.JoinHostPort(masterIP, strconv.Itoa(int(port)))
	return hostPort, nil
}
